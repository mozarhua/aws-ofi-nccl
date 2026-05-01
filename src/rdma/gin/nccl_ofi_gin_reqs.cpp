/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_reqs.h"
#include "rdma/gin/nccl_ofi_gin_resources.h"
#include "nccl_ofi_tracepoint.h"

int nccl_net_ofi_gin_op_req_t::op_req_ctx::handle_cq_entry(struct fi_cq_entry *cq_entry_base,
							   fi_addr_t src_addr, uint16_t rail_id)
{
	nccl_net_ofi_gin_op_req_t *req = cpp_container_of(this, &nccl_net_ofi_gin_op_req_t::ctx);
	return req->handle_cq_entry(cq_entry_base, src_addr, rail_id);
}

int nccl_net_ofi_gin_op_req_t::op_req_ctx::handle_error_entry(struct fid_cq *cq,
							      struct fi_cq_err_entry *err_entry,
							      uint16_t rail_id)
{
	int ret = 0;

	if (err_entry->err == FI_ECANCELED) {
		/* Closing an EP with posted receives will (erroneously) generate
		   cancellation events for the posted receives with the EFA provider
		   in Libfabric versions prior to 1.22. These events are harmless
		   and can be ignored.

		   With Libfabric 1.22 and later, we shouldn't get these cancel
		   events at all. The plugin does not explicitly call fi_cancel. */
		return 0;
	}

	nccl_net_ofi_gin_op_req_t *req = cpp_container_of(this, &nccl_net_ofi_gin_op_req_t::ctx);

	NCCL_OFI_WARN(
		"Request %p completed with error. RC: %d. Flags: %ld. Error: %d (%s). Completed length: %ld",
		req, err_entry->err, err_entry->flags, err_entry->prov_errno,
		fi_cq_strerror(cq, err_entry->prov_errno, err_entry->err_data, NULL, 0),
		(long)err_entry->len);

	/*
	 * Libfabric error codes directly map to ISO C errno values for standard
	 * error codes up to FI_ERRNO_OFFSET, and libfabric-specific error codes
	 * beyond. nccl_net_ofi_retval_translate() will figure out how to deal
	 * with these, so it is safe to pass up the err with its sign negated.
	 * However, any special-handling for prov_errno should be handled here.
	 */
	ret = -(err_entry->err);
	return ret;
}

/* Receive requests */
nccl_net_ofi_gin_recv_req_t::nccl_net_ofi_gin_recv_req_t(nccl_ofi_gin_resources &resources_arg,
							 nccl_ofi_gin_ep_rail_t &rail_arg)
    : nccl_net_ofi_gin_op_req_t(), resources(resources_arg), rail(rail_arg)
{
	rx_buff_elem = resources.get_rx_buff_fl()->entry_alloc();
	if (!rx_buff_elem) {
		NCCL_OFI_WARN("Failed to allocate rx buffer freelist entry");
		throw std::runtime_error("Failed to allocate rx buffer freelist entry");
	}
}

nccl_net_ofi_gin_recv_req_t::~nccl_net_ofi_gin_recv_req_t()
{
	resources.get_rx_buff_fl()->entry_free(rx_buff_elem);
}

int nccl_net_ofi_gin_recv_req_t::handle_cq_entry(struct fi_cq_entry *cq_entry_base,
						 fi_addr_t src_addr, uint16_t rail_id_arg)
{
	assert(this->rail.rail_id == rail_id_arg);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_RQ_ACK)
	// 0430 redo afer recv_req_t::handle_cq_entry (to know what roughly jumping from the parent / container took).
	//auto *gin_comm2 = resources.get_ep().get_profile_comm();
	//if (gin_comm2 && gin_comm2->histogram_recording && gin_comm2->hist_progress) {
	//	gin_comm2->hist_progress->stop_timer();
	//}
#endif
	auto *cq_entry = reinterpret_cast<struct fi_cq_data_entry *>(cq_entry_base);

	int ret = 0;

	if (cq_entry->flags & FI_REMOTE_WRITE) {
		/* RDMA write-immediate completion */
		uint32_t comm_id = GIN_IMM_GET_COMM_ID(cq_entry->data);
#if (PROFILE_GIN_PROGRESS == GIN_PROG_RQ_NON_ACK)
		//auto *gin_comm2 = resources.get_ep().get_profile_comm();
		//if (gin_comm2 && gin_comm2->histogram_recording && gin_comm2->hist_progress) {
		//	gin_comm2->hist_progress->stop_timer();
		//}
#endif
		auto &gin_comm = resources.get_comm(comm_id);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_RQ_NON_ACK)
		auto *gin_comm2 = resources.get_ep().get_profile_comm();
		if (gin_comm2 && gin_comm2->histogram_recording && gin_comm2->hist_progress) {
			gin_comm2->hist_progress->stop_timer();
		}
#endif
		ret = gin_comm.handle_signal_write_completion(cq_entry, src_addr, rail_id_arg);
		if (ret != 0) {
			NCCL_OFI_WARN("gin_handle_signal_write_completion failure");
			return ret;
		}
	} else {
		/* Dispatch by msg_type — at offset 0 in both message structs. */
		auto msg_type = static_cast<gin_ack_msg_t *>(rx_buff_elem->ptr)->msg_type;
		if (msg_type == GIN_MSG_TYPE_ACK) {
			auto *ack_msg =
				static_cast<gin_ack_msg_t *>(rx_buff_elem->ptr);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_RQ_ACK)
			// 0430 redo before get_comm
			//auto *gin_comm2 = resources.get_ep().get_profile_comm();
			//if (gin_comm2 && gin_comm2->histogram_recording && gin_comm2->hist_progress) {
			//	gin_comm2->hist_progress->stop_timer();
			//}
#endif
			auto &gin_comm = resources.get_comm(ack_msg->ack.comm_id);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_RQ_ACK)
			// 0430 redo afer get_comm
			//auto *gin_comm2 = resources.get_ep().get_profile_comm();
			//if (gin_comm2 && gin_comm2->histogram_recording && gin_comm2->hist_progress) {
			//	gin_comm2->hist_progress->stop_timer();
			//}
#endif
			ret = gin_comm.handle_ack_completion(src_addr, rail_id_arg,
							     ack_msg);
			if (OFI_UNLIKELY(ret != 0)) {
				NCCL_OFI_WARN("handle_ack_completion failure");
				return ret;
			}
#if (PROFILE_GIN_PROGRESS == GIN_PROG_RQ_ACK)
			auto *gin_comm2 = resources.get_ep().get_profile_comm();
			if (gin_comm2 && gin_comm2->histogram_recording && gin_comm2->hist_progress) {
				gin_comm2->hist_progress->stop_timer();
			}
#endif
		} else if (msg_type == GIN_MSG_TYPE_METADATA) {
			auto *msg = static_cast<nccl_net_ofi_gin_signal_metadata_msg_t *>(
				rx_buff_elem->ptr);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_RQ_NON_ACK)
			//auto *gin_comm2 = resources.get_ep().get_profile_comm();
			//if (gin_comm2 && gin_comm2->histogram_recording && gin_comm2->hist_progress) {
			//	gin_comm2->hist_progress->stop_timer();
			//}
#endif
			auto &gin_comm = resources.get_comm(msg->remote_comm_id);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_RQ_NON_ACK)
			auto *gin_comm2 = resources.get_ep().get_profile_comm();
			if (gin_comm2 && gin_comm2->histogram_recording && gin_comm2->hist_progress) {
				gin_comm2->hist_progress->stop_timer();
			}
#endif
			ret = gin_comm.handle_signal_metadata_completion(msg, src_addr,
									 rail_id_arg);
			if (ret != 0) {
				NCCL_OFI_WARN(
					"gin_handle_signal_metadata_completion failure");
				return ret;
			}
		} else {
			NCCL_OFI_WARN("Unknown GIN message type %d", msg_type);
			return -EINVAL;
		}
	}

	/* Repost this req */
	return post_or_add_pending();
}

int nccl_net_ofi_gin_recv_req_t::post()
{
	auto *mr_handle = static_cast<nccl_ofi_gin_mr_handle_t *>(rx_buff_elem->mr_handle);
	struct fid_ep *ofi_ep = rail.ofi_ep.get();
	size_t size = sizeof(nccl_net_ofi_gin_signal_metadata_msg_t);
	void *desc = fi_mr_desc(mr_handle->get_mr(rail.rail_id));

	ssize_t rc = fi_recv(ofi_ep, rx_buff_elem->ptr, size, desc, FI_ADDR_UNSPEC, &ctx.ofi_ctx);
	if (rc != 0 && rc != -FI_EAGAIN) {
		NCCL_OFI_WARN("Failed to post recv. RC: %zd", rc);
	}

	return rc;
}

int nccl_net_ofi_gin_recv_req_t::post_or_add_pending()
{
#if (PROFILE_GIN_PROGRESS == GIN_PROG_CQ_COMMON_2)
	auto *gin_comm = resources.get_ep().get_profile_comm();
	if (gin_comm && gin_comm->histogram_recording && gin_comm->hist_progress) {
		gin_comm->hist_progress->start_timer();
	}
#endif
	int ret = post();
	if (ret == -FI_EAGAIN) {
		resources.add_pending_req(this);
		ret = 0;
	}
#if (PROFILE_GIN_PROGRESS == GIN_PROG_CQ_COMMON_2)
	if (gin_comm && gin_comm->histogram_recording && gin_comm->hist_progress) {
		gin_comm->hist_progress->stop_timer();
	}
#endif

	return ret;
}

int nccl_net_ofi_gin_sendack_req_t::handle_cq_entry(struct fi_cq_entry * /*cq_entry_base*/,
						     fi_addr_t /*src_addr*/,
						     uint16_t /*rail_id_arg*/)
{
#if (PROFILE_GIN_PROGRESS == GIN_PROG_SQ_COMP)
	//if (gin_comm.histogram_recording && gin_comm.hist_progress) {
	//	gin_comm.hist_progress->stop_timer();
	//}
#endif
	gin_comm.decrement_outstanding_ack_counter();

	gin_comm.get_resources().return_req_to_pool(this);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_SQ_COMP)
	//if (gin_comm.histogram_recording && gin_comm.hist_progress) {
	//	gin_comm.hist_progress->stop_timer();
	//}
#endif
	return 0;
}

int nccl_net_ofi_gin_sendack_req_t::post()
{
	auto *ack_handle =
		static_cast<nccl_ofi_gin_mr_handle_t *>(ack_elem->mr_handle);

	ssize_t rc = fi_send(ep, ack_elem->ptr, sizeof(gin_ack_msg_t),
			     fi_mr_desc(ack_handle->get_mr(rail_id)), remote_addr,
			     &ctx.ofi_ctx);

	if (rc != 0 && rc != -FI_EAGAIN) {
		NCCL_OFI_WARN("Failed to post ACK send. RC: %zd", rc);
	} else if (rc == 0) {
		gin_comm.increment_outstanding_ack_counter();
	}

	return rc;
}

nccl_net_ofi_gin_sendack_req_t::~nccl_net_ofi_gin_sendack_req_t()
{
	ack_fl->entry_free(ack_elem);
}

int nccl_ofi_rdma_gin_iputsignal_req::test(int *done)
{
#if (PROFILE_GIN_PROGRESS == GIN_PROG_E2E)
	int rec = gin_comm.histogram_recording;
	if (rec && gin_comm.hist_test)
		gin_comm.hist_test->start_timer();
#endif
	if (OFI_UNLIKELY(any_reqs_pending == 0)) {
		auto &gin_ep = gin_comm.get_resources().get_ep();
		if (is_ack_requested) {
			/* This message requested ACK (SIGNAL, PUT-SIGNAL, or every Nth PUT) */
			bool ack_outstanding = gin_comm.query_ack_outstanding(peer_rank, msg_seq_num);
			*done = !ack_outstanding;
		} else {
			/* This message doesn't need ACK (most PUTs) */
			*done = 1;
		}
		if (OFI_UNLIKELY(*done)) {
			std::lock_guard scoped_ep_lock(gin_ep.ep_lock);
			NCCL_OFI_TRACE(NCCL_NET, "Completed iputSignal seq num %hu on initiator",
				       this->msg_seq_num);
			NCCL_OFI_TRACE_GIN_IPUT_SIGNAL_END(gin_comm.get_dev(), &gin_comm, peer_rank,
							   msg_seq_num, this);
			gin_comm.get_resources().return_req_to_pool(this);
		}
	} else {
		//*done = 0;
	}


	/* If not done, the GIN plugin will do nothing.

	   The analogous test() call in net plugin code will progress the CQ
	   here. However, for GIN, given NCCL's current usage, this isn't
	   necessary. The GIN API has a separate ginProgress call, and NCCL's
	   progress thread will continually call `ginProgress` anyway. */

#if (PROFILE_GIN_PROGRESS == GIN_PROG_E2E)
	if (rec && gin_comm.hist_test)
		gin_comm.hist_test->stop_timer();
#endif
	return 0;
}

int nccl_net_ofi_gin_write_req_t::post()
{
	ssize_t rc =
		fi_writedata(ep, src, size, desc, imm_data, remote_addr, dest, key, &ctx.ofi_ctx);

	if (OFI_UNLIKELY(rc != 0 && rc != -FI_EAGAIN)) {
		NCCL_OFI_WARN("Failed call to fi_writedata; RC: %zd", rc);
	}

	return rc;
}

int nccl_net_ofi_gin_write_req_t::handle_cq_entry(struct fi_cq_entry * /*cq_entry_base*/,
						  fi_addr_t /*src_addr*/, uint16_t rail_id)
{
#if (PROFILE_GIN_PROGRESS == GIN_PROG_SQ_COMP)
	auto *gin_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(comm);
	if (gin_comm->histogram_recording && gin_comm->hist_progress) {
		gin_comm->hist_progress->stop_timer();
	}
#endif
	NCCL_OFI_TRACE_GIN_WRITE_END(dev, rail_id, comm, rank, msg_seq_num, this);

	assert(req_pending_byte != nullptr);
	*req_pending_byte = false;

	//auto *gin_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(comm);
	gin_comm->get_resources().return_req_to_pool(this);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_SQ_COMP)
	//if (gin_comm->histogram_recording && gin_comm->hist_progress) {
	//	gin_comm->hist_progress->stop_timer();
	//}
#endif
	return 0;
}

int nccl_net_ofi_gin_metadata_send_req_t::post()
{
	nccl_ofi_gin_mr_handle_t *metadata_handle =
		static_cast<nccl_ofi_gin_mr_handle_t *>(metadata_elem->mr_handle);

	ssize_t rc =
		fi_send(ep, metadata_elem->ptr, sizeof(nccl_net_ofi_gin_signal_metadata_msg_t),
			fi_mr_desc(metadata_handle->get_mr(rail_id)), remote_addr, &ctx.ofi_ctx);
	if (rc != 0 && rc != -FI_EAGAIN) {
		NCCL_OFI_WARN("fi_send failed with RC %zd", rc);
	}

	return rc;
}

int nccl_net_ofi_gin_metadata_send_req_t::handle_cq_entry(struct fi_cq_entry * /*cq_entry_base*/,
							  fi_addr_t /*src_addr*/,
							  uint16_t rail_id_arg)
{
#if (PROFILE_GIN_PROGRESS == GIN_PROG_SQ_COMP)
	auto *gin_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(comm);
	if (gin_comm->histogram_recording && gin_comm->hist_progress) {
		gin_comm->hist_progress->stop_timer();
	}
#endif
	NCCL_OFI_TRACE_GIN_METADATA_SEND_END(dev, rail_id_arg, comm, rank, msg_seq_num, this);

	assert(req_pending_byte != nullptr);
	*req_pending_byte = false;

	//auto *gin_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(comm);
	gin_comm->get_resources().return_req_to_pool(this);

#if (PROFILE_GIN_PROGRESS == GIN_PROG_SQ_COMP)
	if (gin_comm->histogram_recording && gin_comm->hist_progress) {
		gin_comm->hist_progress->stop_timer();
	}
#endif
	return 0;
}

nccl_net_ofi_gin_metadata_send_req_t::~nccl_net_ofi_gin_metadata_send_req_t()
{
	metadata_fl->entry_free(metadata_elem);
}
