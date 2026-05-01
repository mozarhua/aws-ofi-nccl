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
		bins(binner.get_num_bins()), residues(binner.get_num_bins()), aux_counts(binner.get_num_bins()),
		overhead(ovh), num_samples(0), first_insert(true)
	{
	}

	void insert(const T& input_val, size_t extra_counter = 0)
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
		aux_counts[binner_index] += extra_counter;
		num_samples++;
	}

	void print_stats(void) {
		auto range_labels = binner.get_bin_ranges();
		// skip min / max prints where no samples.
		if (num_samples == 0) {
			min_val = 0;
			max_val = 0;
		}
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
			ss  << "\t    aux: ";
			if (aux_counts[i] > 0) {
				ss  << (float) aux_counts[i] / (float) bins[i];
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
	std::vector<std::size_t> aux_counts;
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
// DuraUnit can be defined as std::chrono::microseconds, nanoseconds, etc.
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

	timer_histogram(const std::string &description_arg, Binner binner_arg, DuraUnit ovh = DuraUnit::zero())
		: histogram<T, Binner>(description_arg, binner_arg, ovh.count())
	{
		this->overhead = ovh;
	}

	void start_timer(void)
	{
		start_time = clock::now();
		timing = true;
		asm volatile ("" : : : "memory");
	}

	rep stop_timer(int extra_counter = 0)
	{
		asm volatile ("" : : : "memory");
		if (!timing)
			return 0;
		timing = false;
		auto now = clock::now();
		auto duration = std::chrono::duration_cast<DuraUnit>(now - start_time);
		auto val = (duration > overhead) ? (duration - overhead).count() : 0;
		insert(val, extra_counter);
		return val;
	}


protected:
	typename clock::time_point start_time;
	DuraUnit overhead = DuraUnit::zero();
	bool timing = false;
};


#define GIN_PROG_E2E (1)
#define GIN_PROG_CQ_COMMON_0 (2)	// ep.lock + beginning of for loop
#define GIN_PROG_CQ_COMMON_1 (3)	// fi_cq_readfrom
#define GIN_PROG_CQ_COMMON_2 (4)	// posting rx buffer back
#define GIN_PROG_SQ_COMP (5)
#define GIN_PROG_RQ_NON_ACK (6)
#define GIN_PROG_RQ_ACK (7)
#define GIN_PROG_SEND_ACK (8)
#define GIN_PROG_SIGNAL (9)

//#define PROFILE_GIN_PROGRESS (GIN_PROG_CQ_COMMON_0)
//#define PROFILE_GIN_PROGRESS (GIN_PROG_RQ_NON_ACK)
#define PROFILE_GIN_PROGRESS (GIN_PROG_SQ_COMP)



#endif
