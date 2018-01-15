from __future__ import print_function

import mmap
import os
import sys
import time

import posix_ipc as ipc
import numpy as np

from nn import TheanoLZN
from stats_timer import StatsTimer

# TODO make these command line options
DISPLAY_TIMING_INFO = True
PLAYOUTS_PER_MOVE = 1600
MOVES_PER_GAME_AVERAGE = 350

PRINT_EVERY_SECONDS = 200 # every 200 seconds ~ once per game

BOARD_SIZE = 19
BOARD_SQUARES = BOARD_SIZE ** 2
INPUT_CHANNELS = 18
SIZE_OF_FLOAT = 4
SIZE_OF_INT = 4

# prob of each move + prob of pass + eval of board position
OUTPUT_PREDICTIONS = BOARD_SQUARES + 2

INSTANCE_INPUTS  = INPUT_CHANNELS * BOARD_SQUARES
INSTANCE_OUTPUTS = OUTPUT_PREDICTIONS


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
    total_input_size  = num_instances * INSTANCE_INPUTS * SIZE_OF_FLOAT
    total_output_size = num_instances * INSTANCE_OUTPUTS * SIZE_OF_FLOAT

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
    counter    = mv[:counter_size].view(np.int32)
    input_mem  = mv[counter_size:counter_size + total_input_size].view(np.float32)
    output_mem = mv[counter_size + total_input_size:].view(np.float32)

    # reset all shared memory
    mv[:] = 0

    # create views for input_mem, output_mem for each instance
    inputs = np.split(input_mem, num_instances)
    outputs = np.split(output_mem, num_instances)
    assert len(inputs[0]) == INSTANCE_INPUTS
    assert len(outputs[0]) == INSTANCE_OUTPUTS

    return counter, inputs, outputs


def getReadyInstanceData(smpB, shared_input, batch_size, stats_timer):
    stats_timer.start()

    instance_ids = []
    input_data = []
    checks = 0
    while len(instance_ids) < batch_size:
        for instance_id, input_ready_smp in enumerate(smpB):
            # not ready to acquire
            if input_ready_smp.value <= 0:
                continue

            smpB[instance_id].acquire()

            instance_ids.append(instance_id)
            input_data.append(shared_input[instance_id])

            if len(instance_ids) == batch_size:
                break

        checks += 1
        if checks % 10 == 9:
            # sleep a tiny fraction of second to help with CPU usage
            # min sleep time is ~50us so only do this every couple of loops
            time.sleep(1e-5)


    data = np.concatenate(input_data)
    stats_timer.stop()

    return instance_ids, data


def runNN(net, instance_ids, input_data, output_mem, smpA, stats_timer):
    stats_timer.start()

    out = net.runNN(input_data)
    assert out.shape == (len(instance_ids), INSTANCE_OUTPUTS)

    for i, instance_id in enumerate(instance_ids):
        output_mem[instance_id][:] = out[i]

        smpA[instance_id].release() # send result to client

    stats_timer.stop()


def printStats(t0, minibatches, batch_size, stat_timers):
    total_time_min = (time.perf_counter() - t0) / 60.0
    playouts = minibatches * batch_size
    moves = playouts // PLAYOUTS_PER_MOVE
    games = moves / MOVES_PER_GAME_AVERAGE
    print("\n\t({:.1f} minutes) minibatch iteration {}x{} = {} playouts ~= {} moves ~= {:.1f} games"
        .format(total_time_min, minibatches, batch_size, playouts, moves, games))

    for stat_timer in stat_timers:
        print("\t" + stat_timer.getSummary())
    print()


def main():
    leename = os.environ.get("LEELAZ", "lee")
    print("Using batch name: ", leename)

    num_instances = int(sys.argv[1])
    batch_size = int(sys.argv[2])               # real batch size

    print("%d instances using batch size %d" % (num_instances, batch_size))

    counter, inputs, output_mem = setupMemory(leename, num_instances)
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
    t0 = time.perf_counter()
    last_status = 0

    print("Waiting for %d autogtp instances to run" % num_instances)

    minibatches_run = 0
    while True:
        instance_ids, batch_data = \
            getReadyInstanceData(smpB, inputs, batch_size, get_data_timing)

        runNN(net, instance_ids, batch_data, output_mem, smpA, nn_timing)

        minibatches_run += 1

        # Limit how often this check is performed
        if minibatches_run % 100 == 0:
            t = time.perf_counter()
            if (t - last_status) > PRINT_EVERY_SECONDS:
                last_status = t
                printStats(t0, minibatches_run, batch_size, [get_data_timing, nn_timing])


if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("Usage: %s num-instances batch-size" % sys.argv[0])
        sys.exit(-1)

    main()
