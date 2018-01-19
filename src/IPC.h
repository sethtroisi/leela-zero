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

#ifndef IPC_H_INCLUDED
#define IPC_H_INCLUDED

#include "config.h"

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <vector>

using namespace boost::interprocess;


class IPC {
public:
    IPC();
    ~IPC();

    void getResult(
        const std::vector<float>& input,
        std::vector<float>& output,
        float& policy_out);

private:
    shared_memory_object m_sh_mem;
    mapped_region m_region;

    int m_instance_id;
    int* m_metadata_mem;
    float* m_input_mem;
    float* m_result_mem;
    std::unique_ptr<named_semaphore> m_input_ready_sem;
    std::unique_ptr<named_semaphore> m_output_ready_sem;
};

#endif
