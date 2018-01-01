import mmap
import os
import sys
import time

import posix_ipc as ipc
import numpy as np

from nn import TheanoLZN
from stats_timer import StatsTimer

 # TODO make this a command line option
DISPLAY_TIMING_INFO = True

BOARD_SIZE = 19
BOARD_SQUARES = BOARD_SIZE ** 2
INPUT_CHANNELS = 18
SIZE_OF_FLOAT = 4
SIZE_OF_INT = 4

# prob of each move + prob of pass + eval of board position
OUTPUT_PREDICTIONS = BOARD_SQUARES + 2

INSTANCE_INPUTS       = INPUT_CHANNELS * BOARD_SQUARES
INSTANCE_INPUT_SIZE   = SIZE_OF_FLOAT * INSTANCE_INPUTS
INSTANCE_OUTPUT_SIZE  = SIZE_OF_FLOAT * OUTPUT_PREDICTIONS


def roundup(size, page_size):
    return size + size * (size % page_size > 0)


def createSMP(name):
    smp = ipc.Semaphore(name, ipc.O_CREAT)
    # Unlink semaphore so it deletes when the program exits
    smp.unlink()
    return ipc.Semaphore(name, ipc.O_CREAT)


def createCounters(name, num_instances):
    smp_counter =  createSMP("/%s_counter" % name) # counter semaphore

    smpA = []
    smpB = []
    for i in range(num_instances):
        smpA.append(createSMP("/%s_A_%d" % (name,i)))    # two semaphores for each instance
        smpB.append(createSMP("/%s_B_%d" % (name,i)))

    return smp_counter, smpA, smpB


def setupMemory(leename, num_instances):
    mem_name = "/sm" + leename  # shared memory name
    # 2 counters + num_instance * (input + output)
    counter_size = 2 * SIZE_OF_INT
    total_input_size  = num_instances * INSTANCE_INPUT_SIZE
    total_output_size = num_instances * INSTANCE_OUTPUT_SIZE

    needed_memory_size = counter_size + total_input_size + total_output_size
    shared_memory_size = roundup(needed_memory_size, ipc.PAGE_SIZE)

    try:
        sm = ipc.SharedMemory(mem_name, flags=0, size=shared_memory_size)
    except Exception as ex:
        sm = ipc.SharedMemory(mem_name, flags=ipc.O_CREAT, size=shared_memory_size)

    # memory layout of the shared memory:
    # | counter counter | id id | input 1 | input 2 | .... |  output 1 | output 2| ..... |
    mem = mmap.mmap(sm.fd, sm.size)
    sm.close_fd()

    # Set up aliased names for the shared memory
    mv  = np.frombuffer(mem, dtype=np.uint8, count=needed_memory_size);
    counter    = mv[:counter_size].view(dtype=np.int32)
    input_mem  = mv[counter_size:counter_size + total_input_size].view(dtype=np.float32)
    output_mem = mv[counter_size + total_input_size:]

    # reset all shared memory
    mv[:] = 0

    return counter, input_mem, output_mem


def getReadyInstanceData(smpB, shared_input, batch_size, stats_timer):
    stats_timer.start()

    instance_ids = []
    input_data = []
    while len(instance_ids) < batch_size:
        for instance_id, input_ready_smp in enumerate(smpB):
            # not ready to acquire
            if input_ready_smp.value <= 0:
                continue

            smpB[instance_id].acquire()

            start_data = instance_id * INSTANCE_INPUTS
            end_data = start_data + INSTANCE_INPUTS

            instance_ids.append(instance_id)
            input_data.append(shared_input[start_data : end_data])

            if len(instance_ids) == batch_size:
                break

        # sleep a tiny fraction of second to help with CPU usage
        time.sleep(1e-5)

    data = np.concatenate(input_data)
    stats_timer.stop()

    return instance_ids, data


def runNN(net, instance_ids, input_data, memout, smpA, stats_timer):
    stats_timer.start()

    qqq = net.runNN(input_data)
    sss = qqq.view(dtype = np.uint8)

    for i, instance_id in enumerate(instance_ids):
        memout[instance_id * INSTANCE_OUTPUT_SIZE:
              (instance_id + 1) * INSTANCE_OUTPUT_SIZE] = sss[i]

        smpA[instance_id].release() # send result to client

    stats_timer.stop()


def printStats(t0, minibatches, batch_size, stat_timers):
    print("\n\tminibatch iteration {}x{} = {} playouts".format(
        minibatches, batch_size, minibatches * batch_size))
    for stat_timer in stat_timers:
        print("\t" + stat_timer.getSummary())
    print("\n")


def main():
    leename = os.environ.get("LEELAZ", "lee")
    print("Using batch name: ", leename)

    num_instances = int(sys.argv[1])
    batch_size = int(sys.argv[2])               # real batch size

    if num_instances % batch_size != 0:
        print("Error: number of instances isn't divisible by batch size")
        sys.exit(-1)
    else:
        print("%d instances using batch size %d" % (num_instances, batch_size))

    counter, input_mem, output_mem = setupMemory(leename, num_instances)
    smp_counter, smpA, smpB = createCounters(leename, num_instances)

    # set up counters as [num_instances, next available id]
    counter[:] = [num_instances, 0]

    smp_counter.release() # now clients can take this semaphore

    net = TheanoLZN(batch_size)

    # start a thread to watch for new weights.
    net.startWeightUpdater()

    # set up timers for performance data
    get_data_timing = StatsTimer("collecting data")
    nn_timing = StatsTimer("running net")

    print("Waiting for %d autogtp instances to run" % num_instances)
    t0 = time.perf_counter()

    minibatches_run = 0
    while True:
        instance_ids, batch_data = \
            getReadyInstanceData(smpB, input_mem, batch_size, get_data_timing)

        runNN(net, instance_ids, batch_data, output_mem, smpA, nn_timing)

        minibatches_run += 1
        if minibatches_run % 1000 == 1:
            printStats(t0, minibatches_run, batch_size, [get_data_timing, nn_timing])


if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("Usage: %s num-instances batch-size" % sys.argv[0])
        sys.exit(-1)

    main()
