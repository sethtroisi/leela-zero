/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Seth Troisi

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "config.h"
#include "IPC.h"

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <cassert>
#include <cstdlib>
#include <memory>

#include "Utils.h"


IPC::IPC(void) {
    auto env_name = getenv("LEELAZ");
    std::string name(env_name == nullptr ? "lee" : env_name);

    Utils::myprintf("Initializing IPC \"%s\"\n", name.c_str());

    std::string shared_mem_name = "sm_" + name;
    m_sh_mem = shared_memory_object(open_only, shared_mem_name.c_str(), read_write);
    m_region = mapped_region(m_sh_mem, read_write);

    offset_t size;
    m_sh_mem.get_size(size);
    Utils::myprintf("Shared memory %d bytes\n", size);
    { // get instance id
        int* int_mem = static_cast<int*>(m_region.get_address());
        std::string semaphore_name = name + "_id_sem";
        named_semaphore id_sem{open_only, semaphore_name.c_str()};
        id_sem.wait();

        auto num_instances = int_mem[0];

        // Find an open slot.

        // initial 10 (DEBUG_SIZE) entries
        int_mem += 10;
        // each instance is 4 + INPUTS (18*19*19) + OUTPUTS (19*19+2)
        int instance_size = 4 + 18*19*19 + 19*19+2;

        m_instance_id = -1;
        for (int i = 0; i < num_instances; i++) {
            int* test_mem = int_mem + i *  instance_size;
            // Look for unused slot
            if (test_mem[3] == 0) {
                // setup input_mem / result_mem pointers
                m_instance_id = i;
                m_metadata_mem = test_mem;
                // mark slot in use
                m_metadata_mem[3] = 1;

                float* float_mem = reinterpret_cast<float*>(test_mem);
                // 4 metadata fields
                float_mem += 4;
                m_input_mem = float_mem;
                float_mem += 18*19*19;
                m_result_mem = float_mem;

                break;
            }
        }
        Utils::myprintf("Client Id is %d / %d\n", m_instance_id, num_instances);
        assert(0 <= m_instance_id && m_instance_id < num_instances);

        id_sem.post();
    }
    { // setup io ready semaphores
        auto input_ready_name = name + "_input_ready";
        m_input_ready_sem = std::make_unique<named_semaphore>(
            open_only, input_ready_name.c_str());
        auto output_ready_name =
            name + "_output_" + std::to_string(m_instance_id) + "_ready";
        m_output_ready_sem = std::make_unique<named_semaphore>(
            open_only, output_ready_name.c_str());

    }
    Utils::myprintf("IPC setup complete\n");
    // TODO set hash in metadata_mem
}


IPC::~IPC() {
    // hash, input_ready, output_ready, slot_locked
    assert(m_metadata_mem[1] == 0);
    assert(m_metadata_mem[2] == 0);

    // Release slot!
    m_metadata_mem[0] = 0;
    m_metadata_mem[3] = 0;
}


void IPC::getResult(
    const std::vector<float>& input,
    std::vector<float>& output,
    float& policy_out) {
    std::memcpy(m_input_mem, input.data(), input.size() * sizeof(float));

    // hash, input_ready, output_ready
    assert(m_metadata_mem[1] == 0);
    assert(m_metadata_mem[2] == 0);

    // indicate input_data is set in shared_mem
    m_metadata_mem[1] = 1;
    m_input_ready_sem->post();

    // wait for results to be ready
    m_output_ready_sem->wait();
    assert(m_metadata_mem[1] == 0);
    assert(m_metadata_mem[2] == 1);

    // set outputs
    output.clear();
    output.insert(
        std::begin(output),
        m_result_mem,
        m_result_mem + 19 * 19 + 1);
    policy_out = m_result_mem[19*19+1];

    m_metadata_mem[2] = 0;
}
