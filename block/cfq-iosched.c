/*
 *  CFQ, or complete fairness queueing, disk scheduler.
 *
 *  Based on ideas from a previously unfinished io
 *  scheduler (round robin per-process disk scheduling) and Andrea Arcangeli.
 *
 *  Copyright (C) 2003 Jens Axboe <axboe@kernel.dk>
 */
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/blkdev.h>
#include <linux/elevator.h>
#include <linux/jiffies.h>
#include <linux/rbtree.h>
#include <linux/ioprio.h>
#include <linux/blktrace_api.h>
#include "cfq.h"

/*
 * tunables
 */
/* max queue in one round of service */
static const int cfq_quantum = 8;
static const int cfq_fifo_expire[2] = { HZ / 4, HZ / 8 };
/* maximum backwards seek, in KiB */
static const int cfq_back_max = 16 * 1024;
/* penalty of a backwards seek */
static const int cfq_back_penalty = 2;
static const int cfq_slice_sync = HZ / 10;
static int cfq_slice_async = HZ / 25;
static const int cfq_slice_async_rq = 2;
static int cfq_slice_idle = HZ / 125;
static const int cfq_target_latency = HZ * 3/10; /* 300 ms */
static const int cfq_hist_divisor = 4;

/*
 * offset from end of service tree
 */
#define CFQ_IDLE_DELAY		(HZ / 5)

/*
 * below this threshold, we consider thinktime immediate
 */
#define CFQ_MIN_TT		(2)

#define CFQ_SLICE_SCALE		(5)
#define CFQ_HW_QUEUE_MIN	(5)
#define CFQ_SERVICE_SHIFT       12

#define CFQQ_SEEK_THR		(sector_t)(8 * 100)
#define CFQQ_CLOSE_THR		(sector_t)(8 * 1024)
#define CFQQ_SECT_THR_NONROT	(sector_t)(2 * 32)
#define CFQQ_SEEKY(cfqq)	(hweight32(cfqq->seek_history) > 32/8)

#define RQ_CIC(rq)		\
	((struct cfq_io_context *) (rq)->elevator_private)
#define RQ_CFQQ(rq)		(struct cfq_queue *) ((rq)->elevator_private2)
#define RQ_CFQG(rq)		(struct cfq_group *) ((rq)->elevator_private3)

static struct kmem_cache *cfq_pool;
static struct kmem_cache *cfq_ioc_pool;

static DEFINE_PER_CPU(unsigned long, cfq_ioc_count);
static struct completion *ioc_gone;
static DEFINE_SPINLOCK(ioc_gone_lock);

static DEFINE_SPINLOCK(cic_index_lock);
static DEFINE_IDA(cic_index_ida);

#define CFQ_PRIO_LISTS		IOPRIO_BE_NR
#define cfq_class_idle(cfqq)	((cfqq)->ioprio_class == IOPRIO_CLASS_IDLE)
#define cfq_class_rt(cfqq)	((cfqq)->ioprio_class == IOPRIO_CLASS_RT)

#define sample_valid(samples)	((samples) > 80)
#define rb_entry_cfqg(node)	rb_entry((node), struct cfq_group, rb_node)

/*
 * Most of our rbtree usage is for sorting with min extraction, so
 * if we cache the leftmost node we don't have to walk down the tree
 * to find it. Idea borrowed from Ingo Molnars CFS scheduler. We should
 * move this into the elevator for the rq sorting as well.
 */
struct cfq_rb_root {
	struct rb_root rb;
	struct rb_node *left;
	unsigned count;
	unsigned total_weight;
	u64 min_vdisktime;
	struct rb_node *active;
};
#define CFQ_RB_ROOT	(struct cfq_rb_root) { .rb = RB_ROOT, .left = NULL, \
			.count = 0, .min_vdisktime = 0, }

/*
 * Per process-grouping structure
 */
struct cfq_queue {
	/* reference count */
	atomic_t ref;
	/* various state flags, see below */
	unsigned int flags;
	/* parent cfq_data */
	struct cfq_data *cfqd;
	/* service_tree member */
	struct rb_node rb_node;
	/* service_tree key */
	unsigned long rb_key;
	/* prio tree member */
	struct rb_node p_node;
	/* prio tree root we belong to, if any */
	struct rb_root *p_root;
	/* sorted list of pending requests */
	struct rb_root sort_list;
	/* if fifo isn't expired, next request to serve */
	struct request *next_rq;
	/* requests queued in sort_list */
	int queued[2];
	/* currently allocated requests */
	int allocated[2];
	/* fifo list of requests in sort_list */
	struct list_head fifo;

	/* time when queue got scheduled in to dispatch first request. */
	unsigned long dispatch_start;
	unsigned int allocated_slice;
	unsigned int slice_dispatch;
	/* time when first request from queue completed and slice started. */
	unsigned long slice_start;
	unsigned long slice_end;
	long slice_resid;

	/* pending metadata requests */
	int meta_pending;
	/* number of requests that are on the dispatch list or inside driver */
	int dispatched;

	/* io prio of this group */
	unsigned short ioprio, org_ioprio;
	unsigned short ioprio_class, org_ioprio_class;

	pid_t pid;

	u32 seek_history;
	sector_t last_request_pos;

	struct cfq_rb_root *service_tree;
	struct cfq_queue *new_cfqq;
	struct cfq_group *cfqg;
	struct cfq_group *orig_cfqg;
};

/*
 * First index in the service_trees.
 * IDLE is handled separately, so it has negative index
 */
enum wl_prio_t {
	BE_WORKLOAD = 0,
	RT_WORKLOAD = 1,
	IDLE_WORKLOAD = 2,
};

/*
 * Second index in the service_trees.
 */
enum wl_type_t {
	ASYNC_WORKLOAD = 0,
	SYNC_NOIDLE_WORKLOAD = 1,
	SYNC_WORKLOAD = 2
};

/* This is per cgroup per device grouping structure */
struct cfq_group {
	/* group service_tree member */
	struct rb_node rb_node;

	/* group service_tree key */
	u64 vdisktime;
	unsigned int weight;
	bool on_st;

	/* number of cfqq currently on this group */
	int nr_cfqq;

	/* Per group busy queus average. Useful for workload slice calc. */
	unsigned int busy_queues_avg[2];
	/*
	 * rr lists of queues with requests, onle rr for each priority class.
	 * Counts are embedded in the cfq_rb_root
	 */
	struct cfq_rb_root service_trees[2][3];
	struct cfq_rb_root service_tree_idle;

	unsigned long saved_workload_slice;
	enum wl_type_t saved_workload;
	enum wl_prio_t saved_serving_prio;
	struct blkio_group blkg;
#ifdef CONFIG_CFQ_GROUP_IOSCHED
	struct hlist_node cfqd_node;
	atomic_t ref;
#endif
};

/*
 * Per block device queue structure
 */
struct cfq_data {
	struct request_queue *queue;
	/* Root service tree for cfq_groups */
	struct cfq_rb_root grp_service_tree;
	struct cfq_group root_group;

	/*
	 * The priority currently being served
	 */
	enum wl_prio_t serving_prio;
	enum wl_type_t serving_type;
	unsigned long workload_expires;
	struct cfq_group *serving_group;
	bool noidle_tree_requires_idle;

	/*
	 * Each priority tree is sorted by next_request position.  These
	 * trees are used when determining if two or more queues are
	 * interleaving requests (see cfq_close_cooperator).
	 */
	struct rb_root prio_trees[CFQ_PRIO_LISTS];

	unsigned int busy_queues;

	int rq_in_driver;
	int rq_in_flight[2];

	/*
	 * queue-depth detection
	 */
	int rq_queued;
	int hw_tag;
	/*
	 * hw_tag can be
	 * -1 => indeterminate, (cfq will behave as if NCQ is present, to allow better detection)
	 *  1 => NCQ is present (hw_tag_est_depth is the estimated max depth)
	 *  0 => no NCQ
	 */
	int hw_tag_est_depth;
	unsigned int hw_tag_samples;

	/*
	 * idle window management
	 */
	struct timer_list idle_slice_timer;
	struct work_struct unplug_work;

	struct cfq_queue *active_queue;
	struct cfq_io_context *active_cic;

	/*
	 * async queue for each priority case
	 */
	struct cfq_queue *async_cfqq[2][IOPRIO_BE_NR];
	struct cfq_queue *async_idle_cfqq;

	sector_t last_position;

	/*
	 * tunables, see top of file
	 */
	unsigned int cfq_quantum;
	unsigned int cfq_fifo_expire[2];
	unsigned int cfq_back_penalty;
	unsigned int cfq_back_max;
	unsigned int cfq_slice[2];
	unsigned int cfq_slice_async_rq;
	unsigned int cfq_slice_idle;
	unsigned int cfq_latency;
	unsigned int cfq_group_isolation;

	unsigned int cic_index;
	struct list_head cic_list;

	/*
	 * Fallback dummy cfqq for extreme OOM conditions
	 */
	struct cfq_queue oom_cfqq;

	unsigned long last_delayed_sync;

	/* List of cfq groups being managed on this device*/
	struct hlist_head cfqg_list;
	struct rcu_head rcu;
};

static struct cfq_group *cfq_get_next_cfqg(struct cfq_data *cfqd);

static struct cfq_rb_root *service_tree_for(struct cfq_group *cfqg,
					    enum wl_prio_t prio,
					    enum wl_type_t type)
{
	if (!cfqg)
		return NULL;

	if (prio == IDLE_WORKLOAD)
		return &cfqg->service_tree_idle;

	return &cfqg->service_trees[prio][type];
}

enum cfqq_state_flags {
	CFQ_CFQQ_FLAG_on_rr = 0,	/* on round-robin busy list */
	CFQ_CFQQ_FLAG_wait_request,	/* waiting for a request */
	CFQ_CFQQ_FLAG_must_dispatch,	/* must be allowed a dispatch */
	CFQ_CFQQ_FLAG_must_alloc_slice,	/* per-slice must_alloc flag */
	CFQ_CFQQ_FLAG_fifo_expire,	/* FIFO checked in this slice */
	CFQ_CFQQ_FLAG_idle_window,	/* slice idling enabled */
	CFQ_CFQQ_FLAG_prio_changed,	/* task priority has changed */
	CFQ_CFQQ_FLAG_slice_new,	/* no requests dispatched in slice */
	CFQ_CFQQ_FLAG_sync,		/* synchronous queue */
	CFQ_CFQQ_FLAG_coop,		/* cfqq is shared */
	CFQ_CFQQ_FLAG_split_coop,	/* shared cfqq will be splitted */
	CFQ_CFQQ_FLAG_deep,		/* sync cfqq experienced large depth */
	CFQ_CFQQ_FLAG_wait_busy,	/* Waiting for next request */
};

#define CFQ_CFQQ_FNS(name)						\
static inline void cfq_mark_cfqq_##name(struct cfq_queue *cfqq)		\
{									\
	(cfqq)->flags |= (1 << CFQ_CFQQ_FLAG_##name);			\
}									\
static inline void cfq_clear_cfqq_##name(struct cfq_queue *cfqq)	\
{									\
	(cfqq)->flags &= ~(1 << CFQ_CFQQ_FLAG_##name);			\
}									\
static inline int cfq_cfqq_##name(const struct cfq_queue *cfqq)		\
{									\
	return ((cfqq)->flags & (1 << CFQ_CFQQ_FLAG_##name)) != 0;	\
}

CFQ_CFQQ_FNS(on_rr);
CFQ_CFQQ_FNS(wait_request);
CFQ_CFQQ_FNS(must_dispatch);
CFQ_CFQQ_FNS(must_alloc_slice);
CFQ_CFQQ_FNS(fifo_expire);
CFQ_CFQQ_FNS(idle_window);
CFQ_CFQQ_FNS(prio_changed);
CFQ_CFQQ_FNS(slice_new);
CFQ_CFQQ_FNS(sync);
CFQ_CFQQ_FNS(coop);
CFQ_CFQQ_FNS(split_coop);
CFQ_CFQQ_FNS(deep);
CFQ_CFQQ_FNS(wait_busy);
#undef CFQ_CFQQ_FNS

#ifdef CONFIG_CFQ_GROUP_IOSCHED
#define cfq_log_cfqq(cfqd, cfqq, fmt, args...)	\
	blk_add_trace_msg((cfqd)->queue, "cfq%d%c %s " fmt, (cfqq)->pid, \
			cfq_cfqq_sync((cfqq)) ? 'S' : 'A', \
			blkg_path(&(cfqq)->cfqg->blkg), ##args);

#define cfq_log_cfqg(cfqd, cfqg, fmt, args...)				\
	blk_add_trace_msg((cfqd)->queue, "%s " fmt,			\
				blkg_path(&(cfqg)->blkg), ##args);      \

#else
#define cfq_log_cfqq(cfqd, cfqq, fmt, args...)	\
	blk_add_trace_msg((cfqd)->queue, "cfq%d " fmt, (cfqq)->pid, ##args)
#define cfq_log_cfqg(cfqd, cfqg, fmt, args...)		do {} while (0);
#endif
#define cfq_log(cfqd, fmt, args...)	\
	blk_add_trace_msg((cfqd)->queue, "cfq " fmt, ##args)

/* Traverses through cfq group service trees */
#define for_each_cfqg_st(cfqg, i, j, st) \
	for (i = 0; i <= IDLE_WORKLOAD; i++) \
		for (j = 0, st = i < IDLE_WORKLOAD ? &cfqg->service_trees[i][j]\
			: &cfqg->service_tree_idle; \
			(i < IDLE_WORKLOAD && j <= SYNC_WORKLOAD) || \
			(i == IDLE_WORKLOAD && j == 0); \
			j++, st = i < IDLE_WORKLOAD ? \
			&cfqg->service_trees[i][j]: NULL) \


static inline enum wl_prio_t cfqq_prio(struct cfq_queue *cfqq)
{
	if (cfq_class_idle(cfqq))
		return IDLE_WORKLOAD;
	if (cfq_class_rt(cfqq))
		return RT_WORKLOAD;
	return BE_WORKLOAD;
}


static enum wl_type_t cfqq_type(struct cfq_queue *cfqq)
{
	if (!cfq_cfqq_sync(cfqq))
		return ASYNC_WORKLOAD;
	if (!cfq_cfqq_idle_window(cfqq))
		return SYNC_NOIDLE_WORKLOAD;
	return SYNC_WORKLOAD;
}

static inline int cfq_group_busy_queues_wl(enum wl_prio_t wl,
					struct cfq_data *cfqd,
					struct cfq_group *cfqg)
{
	if (wl == IDLE_WORKLOAD)
		return cfqg->service_tree_idle.count;

	return cfqg->service_trees[wl][ASYNC_WORKLOAD].count
		+ cfqg->service_trees[wl][SYNC_NOIDLE_WORKLOAD].count
		+ cfqg->service_trees[wl][SYNC_WORKLOAD].count;
}

static inline int cfqg_busy_async_queues(struct cfq_data *cfqd,
					struct cfq_group *cfqg)
{
	return cfqg->service_trees[RT_WORKLOAD][ASYNC_WORKLOAD].count
		+ cfqg->service_trees[BE_WORKLOAD][ASYNC_WORKLOAD].count;
}

static void cfq_dispatch_insert(struct request_queue *, struct request *);
static struct cfq_queue *cfq_get_queue(struct cfq_data *, bool,
				       struct io_context *, gfp_t);
static struct cfq_io_context *cfq_cic_lookup(struct cfq_data *,
						struct io_context *);

static inline struct cfq_queue *cic_to_cfqq(struct cfq_io_context *cic,
					    bool is_sync)
{
	return cic->cfqq[is_sync];
}

static inline void cic_set_cfqq(struct cfq_io_context *cic,
				struct cfq_queue *cfqq, bool is_sync)
{
	cic->cfqq[is_sync] = cfqq;
}

#define CIC_DEAD_KEY	1ul
#define CIC_DEAD_INDEX_SHIFT	1

static inline void *cfqd_dead_key(struct cfq_data *cfqd)
{
	return (void *)(cfqd->cic_index << CIC_DEAD_INDEX_SHIFT | CIC_DEAD_KEY);
}

static inline struct cfq_data *cic_to_cfqd(struct cfq_io_context *cic)
{
	struct cfq_data *cfqd = cic->key;

	if (unlikely((unsigned long) cfqd & CIC_DEAD_KEY))
		return NULL;

	return cfqd;
}

/*
 * We regard a request as SYNC, if it's either a read or has the SYNC bit
 * set (in which case it could also be direct WRITE).
 */
static inline bool cfq_bio_sync(struct bio *bio)
{
	return bio_data_dir(bio) == READ || bio_rw_flagged(bio, BIO_RW_SYNCIO);
}

/*
 * scheduler run of queue, if there are requests pending and no one in the
 * driver that will restart queueing
 */
static inline void cfq_schedule_dispatch(struct cfq_data *cfqd)
{
	if (cfqd->busy_queues) {
		cfq_log(cfqd, "schedule dispatch");
		kblockd_schedule_work(cfqd->queue, &cfqd->unplug_work);
	}
}

static int cfq_queue_empty(struct request_queue *q)
{
	struct cfq_data *cfqd = q->elevator->elevator_data;

	return !cfqd->rq_queued;
}

/*
 * Scale schedule slice based on io priority. Use the sync time slice only
 * if a queue is marked sync and has sync io queued. A sync queue with async
 * io only, should not get full sync slice length.
 */
static inline int cfq_prio_slice(struct cfq_data *cfqd, bool sync,
				 unsigned short prio)
{
	const int base_slice = cfqd->cfq_slice[sync];

	WARN_ON(prio >= IOPRIO_BE_NR);

	return base_slice + (base_slice/CFQ_SLICE_SCALE * (4 - prio));
}

static inline int
cfq_prio_to_slice(struct cfq_data *cfqd, struct cfq_queue *cfqq)
{
	return cfq_prio_slice(cfqd, cfq_cfqq_sync(cfqq), cfqq->ioprio);
}

static inline u64 cfq_scale_slice(unsigned long delta, struct cfq_group *cfqg)
{
	u64 d = delta << CFQ_SERVICE_SHIFT;

	d = d * BLKIO_WEIGHT_DEFAULT;
	do_div(d, cfqg->weight);
	return d;
}

static inline u64 max_vdisktime(u64 min_vdisktime, u64 vdisktime)
{
	s64 delta = (s64)(vdisktime - min_vdisktime);
	if (delta > 0)
		min_vdisktime = vdisktime;

	return min_vdisktime;
}

static inline u64 min_vdisktime(u64 min_vdisktime, u64 vdisktime)
{
	s64 delta = (s64)(vdisktime - min_vdisktime);
	if (delta < 0)
		min_vdisktime = vdisktime;

	return min_vdisktime;
}

static void update_min_vdisktime(struct cfq_rb_root *st)
{
	u64 vdisktime = st->min_vdisktime;
	struct cfq_group *cfqg;

	if (st->active) {
		cfqg = rb_entry_cfqg(st->active);
		vdisktime = cfqg->vdisktime;
	}

	if (st->left) {
		cfqg = rb_entry_cfqg(st->left);
		vdisktime = min_vdisktime(vdisktime, cfqg->vdisktime);
	}

	st->min_vdisktime = max_vdisktime(st->min_vdisktime, vdisktime);
}

/*
 * get averaged number of queues of RT/BE priority.
 * average is updated, with a formula that gives more weight to higher numbers,
 * to quickly follows sudden increases and decrease slowly
 */

static inline unsigned cfq_group_get_avg_queues(struct cfq_data *cfqd,
					struct cfq_group *cfqg, bool rt)
{
	unsigned min_q, max_q;
	unsigned mult  = cfq_hist_divisor - 1;
	unsigned round = cfq_hist_divisor / 2;
	unsigned busy = cfq_group_busy_queues_wl(rt, cfqd, cfqg);

	min_q = min(cfqg->busy_queues_avg[rt], busy);
	max_q = max(cfqg->busy_queues_avg[rt], busy);
	cfqg->busy_queues_avg[rt] = (mult * max_q + min_q + round) /
		cfq_hist_divisor;
	return cfqg->busy_queues_avg[rt];
}

static inline unsigned
cfq_group_slice(struct cfq_data *cfqd, struct cfq_group *cfqg)
{
	struct cfq_rb_root *st = &cfqd->grp_service_tree;

	return cfq_target_latency * cfqg->weight / st->total_weight;
}

static inline void
cfq_set_prio_slice(struct cfq_data *cfqd, struct cfq_queue *cfqq)
{
	unsigned slice = cfq_prio_to_slice(cfqd, cfqq);
	if (cfqd->cfq_latency) {
		/*
		 * interested queues (we consider only the ones with the same
		 * priority class in the cfq group)
		 */
		unsigned iq = cfq_group_get_avg_queues(cfqd, cfqq->cfqg,
						cfq_class_rt(cfqq));
		unsigned sync_slice = cfqd->cfq_slice[1];
		unsigned expect_latency = sync_slice * iq;
		unsigned group_slice = cfq_group_slice(cfqd, cfqq->cfqg);

		if (expect_latency > group_slice) {
			unsigned base_low_slice = 2 * cfqd->cfq_slice_idle;
			/* scale low_slice according to IO priority
			 * and sync vs async */
			unsigned low_slice =
				min(slice, base_low_slice * slice / sync_slice);
			/* the adapted slice value is scaled to fit all iqs
			 * into the target latency */
			slice = max(slice * group_slice / expect_latency,
				    low_slice);
		}
	}
	cfqq->slice_start = jiffies;
	cfqq->slice_end = jiffies + slice;
	cfqq->allocated_slice = slice;
	cfq_log_cfqq(cfqd, cfqq, "set_slice=%lu", cfqq->slice_end - jiffies);
}

/*
 * We need to wrap this check in cfq_cfqq_slice_new(), since ->slice_end
 * isn't valid until the first request from the dispatch is activated
 * and the slice time set.
 */
static inline bool cfq_slice_used(struct cfq_queue *cfqq)
{
	if (cfq_cfqq_slice_new(cfqq))
		return 0;
	if (time_before(jiffies, cfqq->slice_end))
		return 0;

	return 1;
}

/*
 * Lifted from AS - choose which of rq1 and rq2 that is best served now.
 * We choose the request that is closest to the head right now. Distance
 * behind the head is penalized and only allowed to a certain extent.
 */
static struct request *
cfq_choose_req(struct cfq_data *cfqd, struct request *rq1, struct request *rq2, sector_t last)
{
	sector_t s1, s2, d1 = 0, d2 = 0;
	unsigned long back_max;
#define CFQ_RQ1_WRAP	0x01 /* request 1 wraps */
#define CFQ_RQ2_WRAP	0x02 /* request 2 wraps */
	unsigned wrap = 0; /* bit mask: requests behind the disk head? */

	if (rq1 == NULL || rq1 == rq2)
		return rq2;
	if (rq2 == NULL)
		return rq1;

	if (rq_is_sync(rq1) && !rq_is_sync(rq2))
		return rq1;
	else if (rq_is_sync(rq2) && !rq_is_sync(rq1))
		return rq2;
	if (rq_is_meta(rq1) && !rq_is_meta(rq2))
		return rq1;
	else if (rq_is_meta(rq2) && !rq_is_meta(rq1))
		return rq2;

	s1 = blk_rq_pos(rq1);
	s2 = blk_rq_pos(rq2);

	/*
	 * by definition, 1KiB is 2 sectors
	 */
	back_max = cfqd->cfq_back_max * 2;

	/*
	 * Strict one way elevator _except_ in the case where we allow
	 * short backward seeks which are biased as twice the cost of a
	 * similar forward seek.
	 */
	if (s1 >= last)
		d1 = s1 - last;
	else if (s1 + back_max >= last)
		d1 = (last - s1) * cfqd->cfq_back_penalty;
	else
		wrap |= CFQ_RQ1_WRAP;

	if (s2 >= last)
		d2 = s2 - last;
	else if (s2 + back_max >= last)
		d2 = (last - s2) * cfqd->cfq_back_penalty;
	else
		wrap |= CFQ_RQ2_WRAP;

	/* Found required data */

	/*
	 * By doing switch() on the bit mask "wrap" we avoid having to
	 * check two variables for all permutations: --> faster!
	 */
	switch (wrap) {
	case 0: /* common case for CFQ: rq1 and rq2 not wrapped */
		if (d1 < d2)
			return rq1;
		else if (d2 < d1)
			return rq2;
		else {
			if (s1 >= s2)
				return rq1;
			else
				return rq2;
		}

	case CFQ_RQ2_WRAP:
		return rq1;
	case CFQ_RQ1_WRAP:
		return rq2;
	case (CFQ_RQ1_WRAP|CFQ_RQ2_WRAP): /* both rqs wrapped */
	default:
		/*
		 * Since both rqs are wrapped,
		 * start with the one that's further behind head
		 * (--> only *one* back seek required),
		 * since back seek takes more time than forward.
		 */
		if (s1 <= s2)
			return rq1;
		else
			return rq2;
	}
}

/*
 * The below is leftmost cache rbtree addon
 */
static struct cfq_queue *cfq_rb_first(struct cfq_rb_root *root)
{
	/* Service tree is empty */
	if (!root->count)
		return NULL;

	if (!root->left)
		root->left = rb_first(&root->rb);

	if (root->left)
		return rb_entry(root->left, struct cfq_queue, rb_node);

	return NULL;
}

static struct cfq_group *cfq_rb_first_group(struct cfq_rb_root *root)
{
	if (!root->left)
		root->left = rb_first(&root->rb);

	if (root->left)
		return rb_entry_cfqg(root->left);

	return NULL;
}

static void rb_erase_init(struct rb_node *n, struct rb_root *root)
{
	rb_erase(n, root);
	RB_CLEAR_NODE(n);
}

static void cfq_rb_erase(struct rb_node *n, struct cfq_rb_root *root)
{
	if (root->left == n)
		root->left = NULL;
	rb_erase_init(n, &root->rb);
	--root->count;
}

/*
 * would be nice to take fifo expire time into account as well
 */
static struct request *
cfq_find_next_rq(struct cfq_data *cfqd, struct cfq_queue *cfqq,
		  struct request *last)
{
	struct rb_node *rbnext = rb_next(&last->rb_node);
	struct rb_node *rbprev = rb_prev(&last->rb_node);
	struct request *next = NULL, *prev = NULL;

	BUG_ON(RB_EMPTY_NODE(&last->rb_node));

	if (rbprev)
		prev = rb_entry_rq(rbprev);

	if (rbnext)
		next = rb_entry_rq(rbnext);
	else {
		rbnext = rb_first(&cfqq->sort_list);
		if (rbnext && rbnext != &last->rb_node)
			next = rb_entry_rq(rbnext);
	}

	return cfq_choose_req(cfqd, next, prev, blk_rq_pos(last));
}

static unsigned long cfq_slice_offset(struct cfq_data *cfqd,
				      struct cfq_queue *cfqq)
{
	/*
	 * just an approximation, should be ok.
	 */
	return (cfqq->cfqg->nr_cfqq - 1) * (cfq_prio_slice(cfqd, 1, 0) -
		       cfq_prio_slice(cfqd, cfq_cfqq_sync(cfqq), cfqq->ioprio));
}

static inline s64
cfqg_key(struct cfq_rb_root *st, struct cfq_group *cfqg)
{
	return cfqg->vdisktime - st->min_vdisktime;
}

static void
__cfq_group_service_tree_add(struct cfq_rb_root *st, struct cfq_group *cfqg)
{
	struct rb_node **node = &st->rb.rb_node;
	struct rb_node *parent = NULL;
	struct cfq_group *__cfqg;
	s64 key = cfqg_key(st, cfqg);
	int left = 1;

	while (*node != NULL) {
		parent = *node;
		__cfqg = rb_entry_cfqg(parent);

		if (key < cfqg_key(st, __cfqg))
			node = &parent->rb_left;
		else {
			node = &parent->rb_right;
			left = 0;
		}
	}

	if (left)
		st->left = &cfqg->rb_node;

	rb_link_node(&cfqg->rb_node, parent, node);
	rb_insert_color(&cfqg->rb_node, &st->rb);
}

static void
cfq_group_service_tree_add(struct cfq_data *cfqd, struct cfq_group *cfqg)
{
	struct cfq_rb_root *st = &cfqd->grp_service_tree;
	struct cfq_group *__cfqg;
	struct rb_node *n;

	cfqg->nr_cfqq++;
	if (cfqg->on_st)
		return;

	/*
	 * Currently put the group at the end. Later implement something
	 * so that groups get lesser vtime based on their weights, so that
	 * if group does not loose all if it was not continously backlogged.
	 */
	n = rb_last(&st->rb);
	if (n) {
		__cfqg = rb_entry_cfqg(n);
		cfqg->vdisktime = __cfqg->vdisktime + CFQ_IDLE_DELAY;
	} else
		cfqg->vdisktime = st->min_vdisktime;

	__cfq_group_service_tree_add(st, cfqg);
	cfqg->on_st = true;
	st->total_weight += cfqg->weight;
}

static void
cfq_group_service_tree_del(struct cfq_data *cfqd, struct cfq_group *cfqg)
{
	struct cfq_rb_root *st = &cfqd->grp_service_tree;

	if (st->active == &cfqg->rb_node)
		st->active = NULL;

	BUG_ON(cfqg->nr_cfqq < 1);
	cfqg->nr_cfqq--;

	/* If there are other cfq queues under this group, don't delete it */
	if (cfqg->nr_cfqq)
		return;

	cfq_log_cfqg(cfqd, cfqg, "del_from_rr group");
	cfqg->on_st = false;
	st->total_weight -= cfqg->weight;
	if (!RB_EMPTY_NODE(&cfqg->rb_node))
		cfq_rb_erase(&cfqg->rb_node, st);
	cfqg->saved_workload_slice = 0;
	cfq_blkiocg_update_dequeue_stats(&cfqg->blkg, 1);
}

static inline unsigned int cfq_cfqq_slice_usage(struct cfq_queue *cfqq)
{
	unsigned int slice_used;

	/*
	 * Queue got expired before even a single request completed or
	 * got expired immediately after first request completion.
	 */
	if (!cfqq->slice_start || cfqq->slice_start == jiffies) {
		/*
		 * Also charge the seek time incurred to the group, otherwise
		 * if there are mutiple queues in the group, each can dispatch
		 * a single request on seeky media and cause lots of seek time
		 * and group will never know it.
		 */
		slice_used = max_t(unsigned, (jiffies - cfqq->dispatch_start),
					1);
	} else {
		slice_used = jiffies - cfqq->slice_start;
		if (slice_used > cfqq->allocated_slice)
			slice_used = cfqq->allocated_slice;
	}

	cfq_log_cfqq(cfqq->cfqd, cfqq, "sl_used=%u", slice_used);
	return slice_used;
}

static void cfq_group_served(struct cfq_data *cfqd, struct cfq_group *cfqg,
				struct cfq_queue *cfqq)
{
	struct cfq_rb_root *st = &cfqd->grp_service_tree;
	unsigned int used_sl, charge_sl;
	int nr_sync = cfqg->nr_cfqq - cfqg_busy_async_queues(cfqd, cfqg)
			- cfqg->service_tree_idle.count;

	BUG_ON(nr_sync < 0);
	used_sl = charge_sl = cfq_cfqq_slice_usage(cfqq);

	if (!cfq_cfqq_sync(cfqq) && !nr_sync)
		charge_sl = cfqq->allocated_slice;

	/* Can't update vdisktime while group is on service tree */
	cfq_rb_erase(&cfqg->rb_node, st);
	cfqg->vdisktime += cfq_scale_slice(charge_sl, cfqg);
	__cfq_group_service_tree_add(st, cfqg);

	/* This group is being expired. Save the context */
	if (time_after(cfqd->workload_expires, jiffies)) {
		cfqg->saved_workload_slice = cfqd->workload_expires
						- jiffies;
		cfqg->saved_workload = cfqd->serving_type;
		cfqg->saved_serving_prio = cfqd->serving_prio;
	} else
		cfqg->saved_workload_slice = 0;

	cfq_log_cfqg(cfqd, cfqg, "served: vt=%llu min_vt=%llu", cfqg->vdisktime,
					st->min_vdisktime);
	cfq_blkiocg_update_timeslice_used(&cfqg->blkg, used_sl);
	cfq_blkiocg_set_start_empty_time(&cfqg->blkg);
}

#ifdef CONFIG_CFQ_GROUP_IOSCHED
static inline struct cfq_group *cfqg_of_blkg(struct blkio_group *blkg)
{
	if (blkg)
		return container_of(blkg, struct cfq_group, blkg);
	return NULL;
}

void
cfq_update_blkio_group_weight(struct blkio_group *blkg, unsigned int weight)
{
	cfqg_of_blkg(blkg)->weight = weight;
}

static struct cfq_group *
cfq_find_alloc_cfqg(struct cfq_data *cfqd, struct cgroup *cgroup, int create)
{
	struct blkio_cgroup *blkcg = cgroup_to_blkio_cgroup(cgroup);
	struct cfq_group *cfqg = NULL;
	void *key = cfqd;
	int i, j;
	struct cfq_rb_root *st;
	struct backing_dev_info *bdi = &cfqd->queue->backing_dev_info;
	unsigned int major, minor;

	cfqg = cfqg_of_blkg(blkiocg_lookup_group(blkcg, key));
	if (cfqg && !cfqg->blkg.dev && bdi->dev && dev_name(bdi->dev)) {
		sscanf(dev_name(bdi->dev), "%u:%u", &major, &minor);
		cfqg->blkg.dev = MKDEV(major, minor);
		goto done;
	}
	if (cfqg || !create)
		goto done;

	cfqg = kzalloc_node(sizeof(*cfqg), GFP_ATOMIC, cfqd->queue->node);
	if (!cfqg)
		goto done;

	for_each_cfqg_st(cfqg, i, j, st)
		*st = CFQ_RB_ROOT;
	RB_CLEAR_NODE(&cfqg->rb_node);

	/*
	 * Take the initial reference that will be released on destroy
	 * This can be thought of a joint reference by cgroup and
	 * elevator which will be dropped by either elevator exit
	 * or cgroup deletion path depending on who is exiting first.
	 */
	atomic_set(&cfqg->ref, 1);

	/* Add group onto cgroup list */
	sscanf(dev_name(bdi->dev), "%u:%u", &major, &minor);
	cfq_blkiocg_add_blkio_group(blkcg, &cfqg->blkg, (void *)cfqd,
					MKDEV(major, minor));
	cfqg->weight = blkcg_get_weight(blkcg, cfqg->blkg.dev);

	/* Add group on cfqd list */
	hlist_add_head(&cfqg->cfqd_node, &cfqd->cfqg_list);

done:
	return cfqg;
}

/*
 * Search for the cfq group current task belongs to. If create = 1, then also
 * create the cfq group if it does not exist. request_queue lock must be held.
 */
static struct cfq_group *cfq_get_cfqg(struct cfq_data *cfqd, int create)
{
	struct cgroup *cgroup;
	struct cfq_group *cfqg = NULL;

	rcu_read_lock();
	cgroup = task_cgroup(current, blkio_subsys_id);
	cfqg = cfq_find_alloc_cfqg(cfqd, cgroup, create);
	if (!cfqg && create)
		cfqg = &cfqd->root_group;
	rcu_read_unlock();
	return cfqg;
}

static inline struct cfq_group *cfq_ref_get_cfqg(struct cfq_group *cfqg)
{
	atomic_inc(&cfqg->ref);
	return cfqg;
}

static void cfq_link_cfqq_cfqg(struct cfq_queue *cfqq, struct cfq_group *cfqg)
{
	/* Currently, all async queues are mapped to root group */
	if (!cfq_cfqq_sync(cfqq))
		cfqg = &cfqq->cfqd->root_group;

	cfqq->cfqg = cfqg;
	/* cfqq reference on cfqg */
	atomic_inc(&cfqq->cfqg->ref);
}

static void cfq_put_cfqg(struct cfq_group *cfqg)
{
	struct cfq_rb_root *st;
	int i, j;

	BUG_ON(atomic_read(&cfqg->ref) <= 0);
	if (!atomic_dec_and_test(&cfqg->ref))
		return;
	for_each_cfqg_st(cfqg, i, j, st)
		BUG_ON(!RB_EMPTY_ROOT(&st->rb) || st->active != NULL);
	kfree(cfqg);
}

static void cfq_destroy_cfqg(struct cfq_data *cfqd, struct cfq_group *cfqg)
{
	/* Something wrong if we are trying to remove same group twice */
	BUG_ON(hlist_unhashed(&cfqg->cfqd_node));

	hlist_del_init(&cfqg->cfqd_node);

	/*
	 * Put the reference taken at the time of creation so that when all
	 * queues are gone, group can be destroyed.
	 */
	cfq_put_cfqg(cfqg);
}

static void cfq_release_cfq_groups(struct cfq_data *cfqd)
{
	struct hlist_node *pos, *n;
	struct cfq_group *cfqg;

	hlist_for_each_entry_safe(cfqg, pos, n, &cfqd->cfqg_list, cfqd_node) {
		/*
		 * If cgroup removal path got to blk_group first and removed
		 * it from cgroup list, then it will take care of destroying
		 * cfqg also.
		 */
		if (!cfq_blkiocg_del_blkio_group(&cfqg->blkg))
			cfq_destroy_cfqg(cfqd, cfqg);
	}
}

/*
 * Blk cgroup controller notification saying that blkio_group object is being
 * delinked as associated cgroup object is going away. That also means that
 * no new IO will come in this group. So get rid of this group as soon as
 * any pending IO in the group is finished.
 *
 * This function is called under rcu_read_lock(). key is the rcu protected
 * pointer. That means "key" is a valid cfq_data pointer as long as we are rcu
 * read lock.
 *
 * "key" was fetched from blkio_group under blkio_cgroup->lock. That means
 * it should not be NULL as even if elevator was exiting, cgroup deltion
 * path got to it first.
 */
void cfq_unlink_blkio_group(void *key, struct blkio_group *blkg)
{
	unsigned long  flags;
	struct cfq_data *cfqd = key;

	spin_lock_irqsave(cfqd->queue->queue_lock, flags);
	cfq_destroy_cfqg(cfqd, cfqg_of_blkg(blkg));
	spin_unlock_irqrestore(cfqd->queue->queue_lock, flags);
}

#else /* GROUP_IOSCHED */
static struct cfq_group *cfq_get_cfqg(struct cfq_data *cfqd, int create)
{
	return &cfqd->root_group;
}

static inline struct cfq_group *cfq_ref_get_cfqg(struct cfq_group *cfqg)
{
	return cfqg;
}

static inline void
cfq_link_cfqq_cfqg(struct cfq_queue *cfqq, struct cfq_group *cfqg) {
	cfqq->cfqg = cfqg;
}

static void cfq_release_cfq_groups(struct cfq_data *cfqd) {}
static inline void cfq_put_cfqg(struct cfq_group *cfqg) {}

#endif /* GROUP_IOSCHED */

/*
 * The cfqd->service_trees holds all pending cfq_queue's that have
 * requests waiting to be processed. It is sorted in the order that
 * we will service the queues.
 */
static void cfq_service_tree_add(struct cfq_data *cfqd, struct cfq_queue *cfqq,
				 bool add_front)
{
	struct rb_node **p, *parent;
	struct cfq_queue *__cfqq;
	unsigned long rb_key;
	struct cfq_rb_root *service_tree;
	int left;
	int new_cfqq = 1;
	int group_changed = 0;

#ifdef CONFIG_CFQ_GROUP_IOSCHED
	if (!cfqd->cfq_group_isolation
	    && cfqq_type(cfqq) == SYNC_NOIDLE_WORKLOAD
	    && cfqq->cfqg && cfqq->cfqg != &cfqd->root_group) {
		/* Move this cfq to root group */
		cfq_log_cfqq(cfqd, cfqq, "moving to root group");
		if (!RB_EMPTY_NODE(&cfqq->rb_node))
			cfq_group_service_tree_del(cfqd, cfqq->cfqg);
		cfqq->orig_cfqg = cfqq->cfqg;
		cfqq->cfqg = &cfqd->root_group;
		atomic_inc(&cfqd->root_group.ref);
		group_changed = 1;
	} else if (!cfqd->cfq_group_isolation
		   && cfqq_type(cfqq) == SYNC_WORKLOAD && cfqq->orig_cfqg) {
		/* cfqq is sequential now needs to go to its original group */
		BUG_ON(cfqq->cfqg != &cfqd->root_group);
		if (!RB_EMPTY_NODE(&cfqq->rb_node))
			cfq_group_service_tree_del(cfqd, cfqq->cfqg);
		cfq_put_cfqg(cfqq->cfqg);
		cfqq->cfqg = cfqq->orig_cfqg;
		cfqq->orig_cfqg = NULL;
		group_changed = 1;
		cfq_log_cfqq(cfqd, cfqq, "moved to origin group");
	}
#endif

	service_tree = service_tree_for(cfqq->cfqg, cfqq_prio(cfqq),
						cfqq_type(cfqq));
	if (cfq_class_idle(cfqq)) {
		rb_key = CFQ_IDLE_DELAY;
		parent = rb_last(&service_tree->rb);
		if (parent && parent != &cfqq->rb_node) {
			__cfqq = rb_entry(parent, struct cfq_queue, rb_node);
			rb_key += __cfqq->rb_key;
		} else
			rb_key += jiffies;
	} else if (!add_front) {
		/*
		 * Get our rb key offset. Subtract any residual slice
		 * value carried from last service. A negative resid
		 * count indicates slice overrun, and this should position
		 * the next service time further away in the tree.
		 */
		rb_key = cfq_slice_offset(cfqd, cfqq) + jiffies;
		rb_key -= cfqq->slice_resid;
		cfqq->slice_resid = 0;
	} else {
		rb_key = -HZ;
		__cfqq = cfq_rb_first(service_tree);
		rb_key += __cfqq ? __cfqq->rb_key : jiffies;
	}

	if (!RB_EMPTY_NODE(&cfqq->rb_node)) {
		new_cfqq = 0;
		/*
		 * same position, nothing more to do
		 */
		if (rb_key == cfqq->rb_key &&
		    cfqq->service_tree == service_tree)
			return;

		cfq_rb_erase(&cfqq->rb_node, cfqq->service_tree);
		cfqq->service_tree = NULL;
	}

	left = 1;
	parent = NULL;
	cfqq->service_tree = service_tree;
	p = &service_tree->rb.rb_node;
	while (*p) {
		struct rb_node **n;

		parent = *p;
		__cfqq = rb_entry(parent, struct cfq_queue, rb_node);

		/*
		 * sort by key, that represents service time.
		 */
		if (time_before(rb_key, __cfqq->rb_key))
			n = &(*p)->rb_left;
		else {
			n = &(*p)->rb_right;
			left = 0;
		}

		p = n;
	}

	if (left)
		service_tree->left = &cfqq->rb_node;

	cfqq->rb_key = rb_key;
	rb_link_node(&cfqq->rb_node, parent, p);
	rb_insert_color(&cfqq->rb_node, &service_tree->rb);
	service_tree->count++;
	if ((add_front || !new_cfqq) && !group_changed)
		return;
	cfq_group_service_tree_add(cfqd, cfqq->cfqg);
}

static struct cfq_queue *
cfq_prio_tree_lookup(struct cfq_data *cfqd, struct rb_root *root,
		     sector_t sector, struct rb_node **ret_parent,
		     struct rb_node ***rb_link)
{
	struct rb_node **p, *parent;
	struct cfq_queue *cfqq = NULL;

	parent = NULL;
	p = &root->rb_node;
	while (*p) {
		struct rb_node **n;

		parent = *p;
		cfqq = rb_entry(parent, struct cfq_queue, p_node);

		/*
		 * Sort strictly based on sector.  Smallest to the left,
		 * largest to the right.
		 */
		if (sector > blk_rq_pos(cfqq->next_rq))
			n = &(*p)->rb_right;
		else if (sector < blk_rq_pos(cfqq->next_rq))
			n = &(*p)->rb_left;
		else
			break;
		p = n;
		cfqq = NULL;
	}

	*ret_parent = parent;
	if (rb_link)
		*rb_link = p;
	return cfqq;
}

static void cfq_prio_tree_add(struct cfq_data *cfqd, struct cfq_queue *cfqq)
{
	struct rb_node **p, *parent;
	struct cfq_queue *__cfqq;

	if (cfqq->p_root) {
		rb_erase(&cfqq->p_node, cfqq->p_root);
		cfqq->p_root = NULL;
	}

	if (cfq_class_idle(cfqq))
		return;
	if (!cfqq->next_rq)
		return;

	cfqq->p_root = &cfqd->prio_trees[cfqq->org_ioprio];
	__cfqq = cfq_prio_tree_lookup(cfqd, cfqq->p_root,
				      blk_rq_pos(cfqq->next_rq), &parent, &p);
	if (!__cfqq) {
		rb_link_node(&cfqq->p_node, parent, p);
		rb_insert_color(&cfqq->p_node, cfqq->p_root);
	} else
		cfqq->p_root = NULL;
}

/*
 * Update cfqq's position in the service tree.
 */
static void cfq_resort_rr_list(struct cfq_data *cfqd, struct cfq_queue *cfqq)
{
	/*
	 * Resorting requires the cfqq to be on the RR list already.
	 */
	if (cfq_cfqq_on_rr(cfqq)) {
		cfq_service_tree_add(cfqd, cfqq, 0);
		cfq_prio_tree_add(cfqd, cfqq);
	}
}

/*
 * add to busy list of queues for service, trying to be fair in ordering
 * the pending list according to last request service
 */
static void cfq_add_cfqq_rr(struct cfq_data *cfqd, struct cfq_queue *cfqq)
{
	cfq_log_cfqq(cfqd, cfqq, "add_to_rr");
	BUG_ON(cfq_cfqq_on_rr(cfqq));
	cfq_mark_cfqq_on_rr(cfqq);
	cfqd->busy_queues++;

	cfq_resort_rr_list(cfqd, cfqq);
}

/*
 * Called when the cfqq no longer has requests pending, remove it from
 * the service tree.
 */
static void cfq_del_cfqq_rr(struct cfq_data *cfqd, struct cfq_queue *cfqq)
{
	cfq_log_cfqq(cfqd, cfqq, "del_from_rr");
	BUG_ON(!cfq_cfqq_on_rr(cfqq));
	cfq_clear_cfqq_on_rr(cfqq);

	if (!RB_EMPTY_NODE(&cfqq->rb_node)) {
		cfq_rb_erase(&cfqq->rb_node, cfqq->service_tree);
		cfqq->service_tree = NULL;
	}
	if (cfqq->p_root) {
		rb_erase(&cfqq->p_node, cfqq->p_root);
		cfqq->p_root = NULL;
	}

	cfq_group_service_tree_del(cfqd, cfqq->cfqg);
	BUG_ON(!cfqd->busy_queues);
	cfqd->busy_queues--;
}

/*
 * rb tree support functions
 */
static void cfq_del_rq_rb(struct request *rq)
{
	struct cfq_queue *cfqq = RQ_CFQQ(rq);
	const int sync = rq_is_sync(rq);

	BUG_ON(!cfqq->queued[sync]);
	cfqq->queued[sync]--;

	elv_rb_del(&cfqq->sort_list, rq);

	if (cfq_cfqq_on_rr(cfqq) && RB_EMPTY_ROOT(&cfqq->sort_list)) {
		/*
		 * Queue will be deleted from service tree when we actually
		 * expire it later. Right now just remove it from prio tree
		 * as it is empty.
		 */
		if (cfqq->p_root) {
			rb_erase(&cfqq->p_node, cfqq->p_root);
			cfqq->p_root = NULL;
		}
	}
}

static void cfq_add_rq_rb(struct request *rq)
{
	struct cfq_queue *cfqq = RQ_CFQQ(rq);
	struct cfq_data *cfqd = cfqq->cfqd;
	struct request *__alias, *prev;

	cfqq->queued[rq_is_sync(rq)]++;

	/*
	 * looks a little odd, but the first insert might return an alias.
	 * if that happens, put the alias on the dispatch list
	 */
	while ((__alias = elv_rb_add(&cfqq->sort_list, rq)) != NULL)
		cfq_dispatch_insert(cfqd->queue, __alias);

	if (!cfq_cfqq_on_rr(cfqq))
		cfq_add_cfqq_rr(cfqd, cfqq);

	/*
	 * check if this request is a better next-serve candidate
	 */
	prev = cfqq->next_rq;
	cfqq->next_rq = cfq_choose_req(cfqd, cfqq->next_rq, rq, cfqd->last_position);

	/*
	 * adjust priority tree position, if ->next_rq changes
	 */
	if (prev != cfqq->next_rq)
		cfq_prio_tree_add(cfqd, cfqq);

	BUG_ON(!cfqq->next_rq);
}

static void cfq_reposition_rq_rb(struct cfq_queue *cfqq, struct request *rq)
{
	elv_rb_del(&cfqq->sort_list, rq);
	cfqq->queued[rq_is_sync(rq)]--;
	cfq_blkiocg_update_io_remove_stats(&(RQ_CFQG(rq))->blkg,
					rq_data_dir(rq), rq_is_sync(rq));
	cfq_add_rq_rb(rq);
	cfq_blkiocg_update_io_add_stats(&(RQ_CFQG(rq))->blkg,
			&cfqq->cfqd->serving_group->blkg, rq_data_dir(rq),
			rq_is_sync(rq));
}

static struct request *
cfq_find_rq_fmerge(struct cfq_data *cfqd, struct bio *bio)
{
	struct task_struct *tsk = current;
	struct cfq_io_context *cic;
	struct cfq_queue *cfqq;

	cic = cfq_cic_lookup(cfqd, tsk->io_context);
	if (!cic)
		return NULL;

	cfqq = cic_to_cfqq(cic, cfq_bio_sync(bio));
	if (cfqq) {
		sector_t sector = bio->bi_sector + bio_sectors(bio);

		return elv_rb_find(&cfqq->sort_list, sector);
	}

	return NULL;
}

static void cfq_activate_request(struct request_queue *q, struct request *rq)
{
	struct cfq_data *cfqd = q->elevator->elevator_data;

	cfqd->rq_in_driver++;
	cfq_log_cfqq(cfqd, RQ_CFQQ(rq), "activate rq, drv=%d",
						cfqd->rq_in_driver);

	cfqd->last_position = blk_rq_pos(rq) + blk_rq_sectors(rq);
}

static void cfq_deactivate_request(struct request_queue *q, struct request *rq)
{
	struct cfq_data *cfqd = q->elevator->elevator_data;

	WARN_ON(!cfqd->rq_in_driver);
	cfqd->rq_in_driver--;
	cfq_log_cfqq(cfqd, RQ_CFQQ(rq), "deactivate rq, drv=%d",
						cfqd->rq_in_driver);
}

static void cfq_remove_request(struct request *rq)
{
	struct cfq_queue *cfqq = RQ_CFQQ(rq);

	if (cfqq->next_rq == rq)
		cfqq->next_rq = cfq_find_next_rq(cfqq->cfqd, cfqq, rq);

	list_del_init(&rq->queuelist);
	cfq_del_rq_rb(rq);

	cfqq->cfqd->rq_queued--;
	cfq_blkiocg_update_io_remove_stats(&(RQ_CFQG(rq))->blkg,
					rq_data_dir(rq), rq_is_sync(rq));
	if (rq_is_meta(rq)) {
		WARN_ON(!cfqq->meta_pending);
		cfqq->meta_pending--;
	}
}

static int cfq_merge(struct request_queue *q, struct request **req,
		     struct bio *bio)
{
	struct cfq_data *cfqd = q->elevator->elevator_data;
	struct request *__rq;

	__rq = cfq_find_rq_fmerge(cfqd, bio);
	if (__rq && elv_rq_merge_ok(__rq, bio)) {
		*req = __rq;
		return ELEVATOR_FRONT_MERGE;
	}

	return ELEVATOR_NO_MERGE;
}

static void cfq_merged_request(struct request_queue *q, struct request *req,
			       int type)
{
	if (type == ELEVATOR_FRONT_MERGE) {
		struct cfq_queue *cfqq = RQ_CFQQ(req);

		cfq_reposition_rq_rb(cfqq, req);
	}
}

static void cfq_bio_merged(struct request_queue *q, struct request *req,
				struct bio *bio)
{
	cfq_blkiocg_update_io_merged_stats(&(RQ_CFQG(req))->blkg,
					bio_data_dir(bio), cfq_bio_sync(bio));
}

static void
cfq_merged_requests(struct request_queue *q, struct request *rq,
		    struct request *next)
{
	struct cfq_queue *cfqq = RQ_CFQQ(rq);
	/*
	 * reposition in fifo if next is older than rq
	 */
	if (!list_empty(&rq->queuelist) && !list_empty(&next->queuelist) &&
	    time_before(rq_fifo_time(next), rq_fifo_time(rq))) {
		list_move(&rq->queuelist, &next->queuelist);
		rq_set_fifo_time(rq, rq_fifo_time(next));
	}

	if (cfqq->next_rq == next)
		cfqq->next_rq = rq;
	cfq_remove_request(next);
	cfq_blkiocg_update_io_merged_stats(&(RQ_CFQG(rq))->blkg,
					rq_data_dir(next), rq_is_sync(next));
}

static int cfq_allow_merge(struct request_queue *q, struct request *rq,
			   struct bio *bio)
{
	struct cfq_data *cfqd = q->elevator->elevator_data;
	struct cfq_io_context *cic;
	struct cfq_queue *cfqq;

	/*
	 * Disallow merge of a sync bio into an async request.
	 */
	if (cfq_bio_sync(bio) && !rq_is_sync(rq))
		return false;

	/*
	 * Lookup the cfqq that this bio will be queued with. Allow
	 * merge only if rq is queued there.
	 */
	cic = cfq_cic_lookup(cfqd, current->io_context);
	if (!cic)
		return false;

	cfqq = cic_to_cfqq(cic, cfq_bio_sync(bio));
	return cfqq == RQ_CFQQ(rq);
}

static inline void cfq_del_timer(struct cfq_data *cfqd, struct cfq_queue *cfqq)
{
	del_timer(&cfqd->idle_slice_timer);
	cfq_blkiocg_update_idle_time_stats(&cfqq->cfqg->blkg);
}

static void __cfq_set_active_queue(struct cfq_data *cfqd,
				   struct cfq_queue *cfqq)
{
	if (cfqq) {
		cfq_log_cfqq(cfqd, cfqq, "set_active wl_prio:%d wl_type:%d",
				cfqd->serving_prio, cfqd->serving_type);
		cfq_blkiocg_update_avg_queue_size_stats(&cfqq->cfqg->blkg);
		cfqq->slice_start = 0;
		cfqq->dispatch_start = jiffies;
		cfqq->allocated_slice = 0;
		cfqq->slice_end = 0;
		cfqq->slice_dispatch = 0;

		cfq_clear_cfqq_wait_request(cfqq);
		cfq_clear_cfqq_must_dispatch(cfqq);
		cfq_clear_cfqq_must_alloc_slice(cfqq);
		cfq_clear_cfqq_fifo_expire(cfqq);
		cfq_mark_cfqq_slice_new(cfqq);

		cfq_del_timer(cfqd, cfqq);
	}

	cfqd->active_queue = cfqq;
}

/*
 * current cfqq expired its slice (or was too idle), select new one
 */
static void
__cfq_slice_expired(struct cfq_data *cfqd, struct cfq_queue *cfqq,
		    bool timed_out)
{
	cfq_log_cfqq(cfqd, cfqq, "slice expired t=%d", timed_out);

	if (cfq_cfqq_wait_request(cfqq))
		cfq_del_timer(cfqd, cfqq);

	cfq_clear_cfqq_wait_request(cfqq);
	cfq_clear_cfqq_wait_busy(cfqq);

	/*
	 * If this cfqq is shared between multiple processes, check to
	 * make sure that those processes are still issuing I/Os within
	 * the mean seek distance.  If not, it may be time to break the
	 * queues apart again.
	 */
	if (cfq_cfqq_coop(cfqq) && CFQQ_SEEKY(cfqq))
		cfq_mark_cfqq_split_coop(cfqq);

	/*
	 * store what was left of this slice, if the queue idled/timed out
	 */
	if (timed_out && !cfq_cfqq_slice_new(cfqq)) {
		cfqq->slice_resid = cfqq->slice_end - jiffies;
		cfq_log_cfqq(cfqd, cfqq, "resid=%ld", cfqq->slice_resid);
	}

	cfq_group_served(cfqd, cfqq->cfqg, cfqq);

	if (cfq_cfqq_on_rr(cfqq) && RB_EMPTY_ROOT(&cfqq->sort_list))
		cfq_del_cfqq_rr(cfqd, cfqq);

	cfq_resort_rr_list(cfqd, cfqq);

	if (cfqq == cfqd->active_queue)
		cfqd->active_queue = NULL;

	if (&cfqq->cfqg->rb_node == cfqd->grp_service_tree.active)
		cfqd->grp_service_tree.active = NULL;

	if (cfqd->active_cic) {
		put_io_context(cfqd->active_cic->ioc);
		cfqd->active_cic = NULL;
	}
}

static inline void cfq_slice_expired(struct cfq_data *cfqd, bool timed_out)
{
	struct cfq_queue *cfqq = cfqd->active_queue;

	if (cfqq)
		__cfq_slice_expired(cfqd, cfqq, timed_out);
}

/*
 * Get next queue for service. Unless we have a queue preemption,
 * we'll simply select the first cfqq in the service tree.
 */
static struct cfq_queue *cfq_get_next_queue(struct cfq_data *cfqd)
{
	struct cfq_rb_root *service_tree =
		service_tree_for(cfqd->serving_group, cfqd->serving_prio,
					cfqd->serving_type);

	if (!cfqd->rq_queued)
		return NULL;

	/* There is nothing to dispatch */
	if (!service_tree)
		return NULL;
	if (RB_EMPTY_ROOT(&service_tree->rb))
		return NULL;
	return cfq_rb_first(service_tree);
}

static struct cfq_queue *cfq_get_next_queue_forced(struct cfq_data *cfqd)
{
	struct cfq_group *cfqg;
	struct cfq_queue *cfqq;
	int i, j;
	struct cfq_rb_root *st;

	if (!cfqd->rq_queued)
		return NULL;

	cfqg = cfq_get_next_cfqg(cfqd);
	if (!cfqg)
		return NULL;

	for_each_cfqg_st(cfqg, i, j, st)
		if ((cfqq = cfq_rb_first(st)) != NULL)
			return cfqq;
	return NULL;
}

/*
 * Get and set a new active queue for service.
 */
static struct cfq_queue *cfq_set_active_queue(struct cfq_data *cfqd,
					      struct cfq_queue *cfqq)
{
	if (!cfqq)
		cfqq = cfq_get_next_queue(cfqd);

	__cfq_set_active_queue(cfqd, cfqq);
	return cfqq;
}

static inline sector_t cfq_dist_from_last(struct cfq_data *cfqd,
					  struct request *rq)
{
	if (blk_rq_pos(rq) >= cfqd->last_position)
		return blk_rq_pos(rq) - cfqd->last_position;
	else
		return cfqd->last_position - blk_rq_pos(rq);
}

static inline int cfq_rq_close(struct cfq_data *cfqd, struct cfq_queue *cfqq,
			       struct request *rq)
{
	return cfq_dist_from_last(cfqd, rq) <= CFQQ_CLOSE_THR;
}

static struct cfq_queue *cfqq_close(struct cfq_data *cfqd,
				    struct cfq_queue *cur_cfqq)
{
	struct rb_root *root = &cfqd->prio_trees[cur_cfqq->org_ioprio];
	struct rb_node *parent, *node;
	struct cfq_queue *__cfqq;
	sector_t sector = cfqd->last_position;

	if (RB_EMPTY_ROOT(root))
		return NULL;

	/*
	 * First, if we find a request starting at the end of the last
	 * request, choose it.
	 */
	__cfqq = cfq_prio_tree_lookup(cfqd, root, sector, &parent, NULL);
	if (__cfqq)
		return __cfqq;

	/*
	 * If the exact sector wasn't found, the parent of the NULL leaf
	 * will contain the closest sector.
	 */
	__cfqq = rb_entry(parent, struct cfq_queue, p_node);
	if (cfq_rq_close(cfqd, cur_cfqq, __cfqq->next_rq))
		return __cfqq;

	if (blk_rq_pos(__cfqq->next_rq) < sector)
		node = rb_next(&__cfqq->p_node);
	else
		node = rb_prev(&__cfqq->p_node);
	if (!node)
		return NULL;

	__cfqq = rb_entry(node, struct cfq_queue, p_node);
	if (cfq_rq_close(cfqd, cur_cfqq, __cfqq->next_rq))
		return __cfqq;

	return NULL;
}

/*
 * cfqd - obvious
 * cur_cfqq - passed in so that we don't decide that the current queue is
 * 	      closely cooperating with itself.
 *
 * So, basically we're assuming that that cur_cfqq has dispatched at least
 * one request, and that cfqd->last_position reflects a position on the disk
 * associated with the I/O issued by cur_cfqq.  I'm not sure this is a valid
 * assumption.
 */
static struct cfq_queue *cfq_close_cooperator(struct cfq_data *cfqd,
					      struct cfq_queue *cur_cfqq)
{
	struct cfq_queue *cfqq;

	if (cfq_class_idle(cur_cfqq))
		return NULL;
	if (!cfq_cfqq_sync(cur_cfqq))
		return NULL;
	if (CFQQ_SEEKY(cur_cfqq))
		return NULL;

	/*
	 * Don't search priority tree if it's the only queue in the group.
	 */
	if (cur_cfqq->cfqg->nr_cfqq == 1)
		return NULL;

	/*
	 * We should notice if some of the queues are cooperating, eg
	 * working closely on the same area of the disk. In that case,
	 * we can group them together and don't waste time idling.
	 */
	cfqq = cfqq_close(cfqd, cur_cfqq);
	if (!cfqq)
		return NULL;

	/* If new queue belongs to different cfq_group, don't choose it */
	if (cur_cfqq->cfqg != cfqq->cfqg)
		return NULL;

	/*
	 * It only makes sense to merge sync queues.
	 */
	if (!cfq_cfqq_sync(cfqq))
		return NULL;
	if (CFQQ_SEEKY(cfqq))
		return NULL;

	/*
	 * Do not merge queues of different priority classes
	 */
	if (cfq_class_rt(cfqq) != cfq_class_rt(cur_cfqq))
		return NULL;

	return cfqq;
}

/*
 * Determine whether we should enforce idle window for this queue.
 */

static bool cfq_should_idle(struct cfq_data *cfqd, struct cfq_queue *cfqq)
{
	enum wl_prio_t prio = cfqq_prio(cfqq);
	struct cfq_rb_root *service_tree = cfqq->service_tree;

	BUG_ON(!service_tree);
	BUG_ON(!service_tree->count);

	/* We never do for idle class queues. */
	if (prio == IDLE_WORKLOAD)
		return false;

	/* We do for queues that were marked with idle window flag. */
	if (cfq_cfqq_idle_window(cfqq) &&
	   !(blk_queue_nonrot(cfqd->queue) && cfqd->hw_tag))
		return true;

	/*
	 * Otherwise, we do only if they are the last ones
	 * in their service tree.
	 */
	if (service_tree->count == 1 && cfq_cfqq_sync(cfqq))
		return 1;
	cfq_log_cfqq(cfqd, cfqq, "Not idling. st->count:%d",
			service_tree->count);
	return 0;
}

static void cfq_arm_slice_timer(struct cfq_data *cfqd)
{
	struct cfq_queue *cfqq = cfqd->active_queue;
	struct cfq_io_context *cic;
	unsigned long sl;

	/*
	 * SSD device without seek penalty, disable idling. But only do so
	 * for devices that support queuing, otherwise we still have a problem
	 * with sync vs async workloads.
	 */
	if (blk_queue_nonrot(cfqd->queue) && cfqd->hw_tag)
		return;

	WARN_ON(!RB_EMPTY_ROOT(&cfqq->sort_list));
	WARN_ON(cfq_cfqq_slice_new(cfqq));

	/*
	 * idle is disabled, either manually or by past process history
	 */
	if (!cfqd->cfq_slice_idle || !cfq_should_idle(cfqd, cfqq))
		return;

	/*
	 * still active requests from this queue, don't idle
	 */
	if (cfqq->dispatched)
		return;

	/*
	 * task has exited, don't wait
	 */
	cic = cfqd->active_cic