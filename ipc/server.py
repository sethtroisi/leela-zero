import gc
import mmap
import os
import sys
import time

import posix_ipc as ipc
import numpy as np


BOARD_SIZE = 19
BOARD_SQUARES = BOARD_SIZE ** 2
INPUT_CHANNELS = 18
SIZE_OF_FLOAT = 4
SIZE_OF_INT = 4

# prob of each move + prob of pass + eval of board position
OUTPUT_PREDICTIONS = BOARD_SQUARES + 2

INSTANCE_INPUT_SIZE   = SIZE_OF_FLOAT * INPUT_CHANNELS * BOARD_SQUARES
INSTANCE_OUTPUT_SIZE  = SIZE_OF_FLOAT * OUTPUT_PREDICTIONS


def roundup(size, page_size):
    return size + size * (size % page_size > 0)


def createSMP(name):
    smp = ipc.Semaphore(name, ipc.O_CREAT)
    # Unlink semaphore so it deletes when the program exits
    smp.unlink()

    # TODO can I just return smp???
    return ipc.Semaphore(name, ipc.O_CREAT)


def createCounters(name, num_instances):
    smp_counter =  createSMP("/%s_counter" % name) # counter semaphore

    smpA = []
    smpB = []
    for i in range(num_instances):
        smpA.append(createSMP("/%s_A_%d" % (name,i)))    # two semaphores for each instance
        smpB.append(createSMP("/%s_B_%d" % (name,i)))

    return smp_counter, smpA, smpB

def checkNewNet(nn):
    if not nn.newNetWeight:
        return False
    nn.net = None
    gc.collect()  # hope that GPU memory is freed, not sure :-()
    weights, numBlocks, numFilters = nn.newNetWeight
    print(" %d channels and %d blocks" % (numFilters, numBlocks) )
    nn.net = nn.LZN(weights, numBlocks, numFilters)
    print("...updated weight!")
    nn.newNetWeight = None
    return True

def main():
    leename = os.environ.get("LEELAZ", "lee")
    name = "/sm" + leename  # shared memory name
    print("Using batch name: ", leename)

    num_instances = int(sys.argv[1])
    batch_size = int(sys.argv[2])               # real batch size

    if num_instances % batch_size != 0:
        print("Error: number of instances isn't divisible by batch size")
        sys.exit(-1)
    else:
        print("%d instances using batch size %d" % (num_instances, batch_size))


    #### MEMORY SETUP ####
    # 2 counters + num_instance * (input + output)
    counter_size = 2 * SIZE_OF_INT
    total_input_size  = num_instances * INSTANCE_INPUT_SIZE
    total_output_size = num_instances * INSTANCE_OUTPUT_SIZE

    needed_memory_size = counter_size + total_input_size + total_output_size
    shared_memory_size = roundup(needed_memory_size, ipc.PAGE_SIZE)

    try:
        sm = ipc.SharedMemory(name, flags=0, size=shared_memory_size )
    except Exception as ex:
        sm = ipc.SharedMemory(name, flags=ipc.O_CREAT, size=shared_memory_size )

    # memory layout of the shared memory:
    # | counter counter | id id | input 1 | input 2 | .... |  output 1 | output 2| ..... |
    mem = mmap.mmap(sm.fd, sm.size)
    sm.close_fd()

    # Set up aliased names for the shared memory
    mv  = np.frombuffer(mem, dtype=np.uint8, count=needed_memory_size);
    counter = mv[:counter_size]
    inp     = mv[counter_size:counter_size + total_input_size]
    memout =  mv[counter_size + total_input_size:]

    smp_counter, smpA, smpB = createCounters(leename, num_instances)

    #### NN SETUP ####
    import nn # import our neural network

    # reset everything
    mv[:] = 0

    # batch_size, id
    dt = np.frombuffer(counter, dtype=np.int32, count = 2)
    dt[:] = [num_instances, 0]


    smp_counter.release() # now clients can take this semaphore

    print("Waiting for %d autogtp instances to run" % num_instances)

    net = nn.net

    # t2 = time.perf_counter()
    batches = num_instances // batch_size
    while True:
        for ii in range(batches):
            first_instance = ii * batch_size
            last_instance   = first_instance + batch_size

            # wait for data
            for i in range(batch_size):
                smpB[first_instance + i].acquire()

            # t1 = time.perf_counter()
            # print("delta t1 = ", t1 - t2)
            # t1 = time.perf_counter()

            dt = np.frombuffer(inp[first_instance * INSTANCE_INPUT_SIZE: last_instance * INSTANCE_INPUT_SIZE], dtype=np.float32,
                               count = batch_size * INSTANCE_INPUT_SIZE // SIZE_OF_FLOAT)

            nn.netlock.acquire(True)   # BLOCK HERE
            if checkNewNet(nn):
                net = nn.net
            nn.netlock.release()

            net[0].set_value(dt.reshape( (batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE) ) )

            qqq = net[1]().astype(np.float32)
            ttt = qqq.reshape(batch_size * OUTPUT_PREDICTIONS)
            memout[first_instance * INSTANCE_OUTPUT_SIZE: last_instance * INSTANCE_OUTPUT_SIZE] = ttt.view(dtype=np.uint8)

            for i in range(batch_size):
                smpA[first_instance + i].release() # send result to client

            # t2 = time.perf_counter()
            # print("delta t2 = ", t2- t1)
            # t2 = time.perf_counter()

if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print("Usage: %s num-instances batch-size" % sys.argv[0])
        sys.exit(-1)

    main()
