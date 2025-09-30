//
// Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
//

#ifndef NCCL_OFI_STATS_HISTOGRAM
#define NCCL_OFI_STATS_HISTOGRAM

#include <cassert>
#include <chrono>
#include <cstddef>
#include <string>
#include <sstream>
#include <vector>

#include "nccl_ofi_log.h"
#include "histogram_binner.h"
#include "nccl_ofi_config_bottom.h"

//
// Base histogram class.  Histograms are a lightweight mechanism for tracking
// events occurances in code and are used for instrumenting the plugin code.
//
// T is the type of the data that will be inserted into the histogram.  Any POD
//  will work, and little effort has been put into making the interface safe for
//  non-Pods.
//
template <typename T, typename Binner>
class histogram {
public:
	histogram(const std::string& description_arg, Binner binner_arg, T ovh = 0)
		: description(description_arg), binner(binner_arg),
		bins(binner.get_num_bins()), residues(binner.get_num_bins()), overhead(ovh), num_samples(0), first_insert(true)
	{
	}

	void insert(const T& input_val)
	{
		if (OFI_UNLIKELY(first_insert)) {
			max_val = min_val = input_val;
			first_insert = false;
		}

		if (input_val > max_val) {
			max_val = input_val;
		} else if (input_val < min_val) {
			min_val = input_val;
		}

		std::size_t binner_index = binner.get_bin(input_val);
		bins[binner_index]++;
		residues[binner_index] += input_val;
		num_samples++;
	}

	void print_stats(void) {
		auto range_labels = binner.get_bin_ranges();

		NCCL_OFI_INFO(NCCL_NET, "histogram %s", description.c_str());
		NCCL_OFI_INFO(NCCL_NET, "  min: %ld, max: %ld, num_samples: %lu, overhead_deducted: %ld",
					(long int)min_val, (long int)max_val, num_samples, (long int)overhead);
		for (size_t i = 0 ; i < bins.size() ; ++i) {
			std::stringstream ss;
			ss << "    " << range_labels[i] << " - ";
			if (i + 1 != bins.size()) {
				ss << range_labels[i + 1] - 1;
			} else {
				ss << "    ";
			}
			ss  << "    " << bins[i];
			ss  << "    avg: ";
			if (bins[i] > 0) {
				ss  << residues[i] / bins[i];
			} else {
				ss << 0;
			}
			NCCL_OFI_INFO(NCCL_NET, "%s", ss.str().c_str());
		}
	}

protected:
	std::string description;
	Binner binner;
	std::vector<std::size_t> bins;
	std::vector<T> residues;
	T max_val;
	T min_val;
	T overhead;
	std::size_t num_samples;
	bool first_insert;
};


//
// Histogram class for tracking intervals.  A timer_histogram class can only
// track one interval at a time, and will auto-insert the result when
// stop_timer() is called. Times are recorded in specified unit DuraUnit.
// DuraUnit can be defined as std::chrono::microseconds, naneseconds, etc.
//
// T is the type of the data that will be inserted into the histogram.  Any POD
//  will work, and little effort has been put into making the interface safe for
//  non-Pods.
template <typename Binner, typename clock = std::chrono::steady_clock, typename T = std::size_t,
	  typename DuraUnit = std::chrono::nanoseconds>
class timer_histogram : public histogram<T, Binner> {
public:
	using rep = T;
	using histogram<T, Binner>::insert;

	timer_histogram(const std::string &description_arg, Binner binner_arg, DuraUnit ovh = std::chrono::nanoseconds::zero())
		: histogram<T, Binner>(description_arg, binner_arg, ovh.count())
	{
		this->overhead = ovh;
	}

	void start_timer(void)
	{
		start_time = clock::now();
		asm volatile ("" : : : "memory");
	}

	rep stop_timer(void)
	{
		asm volatile ("" : : : "memory");
		auto now = clock::now();
		auto duration = std::chrono::duration_cast<DuraUnit>(now - start_time);
		duration -= overhead;
		insert(duration.count());
		return duration.count();
	}


protected:
	typename clock::time_point start_time;
	DuraUnit overhead = DuraUnit::zero();
};

// Profiling category base
#define PROF_ISEND_BASE		0x10000
#define PROF_IRECV_BASE		0x20000
#define PROF_TEST_BASE		0x40000

// Send / Recv  sub category
#define PROF_TOTAL		0x0
#define PROF_BEFORE_PENDING_CQ	0x1
#define PROF_PENDING_CQ		0x2
#define PROF_REQ_PREP		0x4
#define PROF_SEND_RECV_PROG	0x8
#define PROF_AFTER_SEND_RECV_PROG	0x10
#define PROF_FI_WRITE		0x20

// Test sub category
#define PROF_TEST_TOTAL		0x0
#define PROF_TEST_PROCESS_CQ	0x1
#define PROF_TEST_CQ_RAIL	0x2
#define PROF_TEST_FI_CQ_READ	0x4

/* Old table for ref...
#define PROF_ISEND_TOTAL	0
#define PROF_ISEND_PENDING_CQ	1
#define PROF_ISEND_SEND_PROG	2
#define PROF_IRECV_TOTAL	3
#define PROF_IRECV_PENDING_CQ	4
#define PROF_IRECV_RECV_PROG	5
#define PROF_TEST_TOTAL		6
#define PROF_TEST_LIBF		7
#define PROF_TEST_CQ_RAIL	8
// Details of plugin
#define PROF_ISEND_BEFORE_PENDING_CQ	9
#define PROF_IRECV_BEFORE_PENDING_CQ	10
#define PROF_TEST_FI_CQ_READ	11
// Send & Recv diving
#define PROF_ISEND_SEND_REQ	12
#define PROF_IRECV_RECV_REQ	13
#define PROF_ISEND_AFTER_PROG	14
#define PROF_IRECV_AFTER_PROG	15
*/

// Alternating following to do different profiling every time.
#define PROF_ISEND	(PROF_ISEND_BASE | PROF_FI_WRITE)
#define PROF_IRECV	(PROF_IRECV_BASE | PROF_FI_WRITE)
#define PROF_TEST	(PROF_TEST_BASE | PROF_TEST_FI_CQ_READ)

#endif
