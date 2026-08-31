/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://www.hdfgroup.org/licenses.               *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/****************/
/* Module Setup */
/****************/

#include "H5SCmodule.h" /* This source code file is part of the H5SC module */
#define H5D_FRIEND      /* Suppress error about including H5Dpkg */

/***********/
/* Headers */
/***********/
#include "H5private.h"   /* Generic Functions            */
#include "H5ACprivate.h" /* Metadata Cache*/
#include "H5Dpkg.h"      /* Datasets                     */
#include "H5Eprivate.h"  /* Error handling               */
#include "H5Fprivate.h"  /* Files                        */
#include "H5MMprivate.h" /* Memory management            */
#include "H5Iprivate.h"
#include "H5VLprivate.h"
#include "H5SCpkg.h" /* Shared chunk cache           */

/****************/
/* Local Macros */
/****************/

/* Initial size of sel_chunks array in H5SC_io_info_t */
#define H5SC_INIT_CHUNK_LIST_SIZE 128

/******************/
/* Local Typedefs */
/******************/

typedef enum H5SC_reclaim_policy_t { H5SC_RECLAIM_ACTIVE = 0, H5SC_RECLAIM_QUIESCENT } H5SC_reclaim_policy_t;

typedef struct H5SC_reclaim_contrib_t {
    size_t clean_bytes;
    size_t dirty_bytes;
} H5SC_reclaim_contrib_t;

/********************/
/* Local Prototypes */
/********************/

/* I/O initialization related functions */
static inline void H5SC__io_sel_chunk_init(H5SC_io_sel_chunk_t *chunk);

static inline H5SC_io_sel_chunk_t *H5SC__io_sel_at(const H5SC_io_info_t *io, size_t relative_idx);

static herr_t H5SC__selection_io_info_init(H5D_t *dset, const H5S_t *file_space, H5SC_io_info_t *sc_io_info);

static herr_t H5SC__build_resident_first_order(H5SC_io_info_t *sc_io_info);

static herr_t H5SC__merge_defined_chunk_selection(H5S_t *defined, const H5S_t *chunk_defined,
                                                  const H5SC_io_sel_chunk_t *sel, unsigned ndims,
                                                  const hsize_t *dset_dims, const hsize_t *chunk_dims);

static herr_t H5SC__io_info_init(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info);
static herr_t H5SC__erase_io_info_init(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space);
static herr_t H5SC__io_info_reset(H5SC_io_info_t *sc_io_info);
static herr_t H5SC__io_info_term(H5SC_io_info_t *sc_io_info);

static void   H5SC__io_scratch_free_storage(H5SC_io_scratch_t *scratch);
static herr_t H5SC__io_scratch_ensure(H5SC_io_info_t *sc_io_info, size_t needed);

static herr_t H5SC__get_defined_chunk(H5D_t *dset, const H5SC_io_sel_chunk_t *sel, H5SC_chunk_t *cached_chk,
                                      bool filtered, bool partial_bound_chunks_different_encoding,
                                      H5S_t **chunk_defined);

static herr_t H5SC__chunk_teardown_resident(H5D_t *dset, H5SC_chunk_t *chunk);

static herr_t H5SC__flush_one_chunk(H5SC_t *cache, H5D_t *dset, H5SC_dset_header_t *dset_hdr,
                                    H5SC_chunk_t *chk);

static herr_t H5SC__evict_one_chunk(H5SC_t *cache, H5D_t *dset, H5SC_dset_header_t *dset_hdr,
                                    H5SC_chunk_t *chk);

static bool H5SC__chunk_outside_extent(const H5D_t *dset, const hsize_t *chunk_dims, const hsize_t *scaled);

static bool H5SC__chunk_is_partial_bound(unsigned ndims, const hsize_t *chunk_dims, const hsize_t *scaled,
                                         const hsize_t *dims);

static bool H5SC__chunk_needs_erase(const H5D_t *dset, const hsize_t *chunk_dims, const hsize_t *scaled,
                                    const hsize_t *old_dims);

static herr_t H5SC__erase_chunk_invalid_extent_region(H5D_t *dset, const hsize_t *chunk_dims,
                                                      const hsize_t *old_dims, H5SC_chunk_t *chk,
                                                      size_t *nbytes, size_t *alloc_size, bool *delete_chunk);

static herr_t H5SC__compute_logical_chunk_index(unsigned ndims, const hsize_t *dset_dims,
                                                const hsize_t *chunk_dims, const hsize_t *elem_coord,
                                                hsize_t *log_chk_idx);

static herr_t H5SC__compute_chunk_key(const haddr_t    *dset_object_header_addr /*in*/,
                                      const hsize_t    *log_chk_coord /*in*/,
                                      H5SC_chunk_key_t *chunk_key /*in,out*/);

/* Cache sizing / eviction helpers */

static herr_t H5SC__account_dset_size_change(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, size_t old_size,
                                             size_t new_size);

static herr_t H5SC__account_chunk_link_change(H5SC_t *cache, H5SC_dset_header_t *dset_hdr,
                                              size_t old_dset_size);

static void H5SC__update_resident_estimate_history(H5SC_dset_header_t *dset_hdr, size_t actual);

/* Enforce cache retention between I/O requests */
static herr_t H5SC__trim_to_quiescent_limit(H5SC_t *cache);

/* Miss-only lookup cache (populate H5SC_io_sel_chunk_t lookup_* fields) */
static herr_t H5SC__lookup_cache_misses(H5D_t *dset, H5SC_io_info_t *sc_io_info);

static herr_t H5SC__lookup_chunk_for_prune(H5D_t *dset, const hsize_t *scaled, haddr_t *addr,
                                           hsize_t *disk_nbytes, bool *exists);

static herr_t H5SC__rekey_dset_chunks_after_extent_change(H5SC_t *cache, H5D_t *dset,
                                                          H5SC_dset_header_t *dset_hdr,
                                                          const hsize_t      *chunk_dims);

static herr_t H5SC__prune_one_scaled_coord_outside_extent(H5SC_t *cache, H5D_t *dset, const hsize_t *scaled);

static herr_t H5SC__for_each_scaled_coord_outside_extent(H5SC_t *cache, H5D_t *dset, unsigned ndims,
                                                         const hsize_t *old_nchunks,
                                                         const hsize_t *new_nchunks);

/* Space-making entry point (enforced against cache->SCC_active_limit during I/O) */
static herr_t H5SC__ensure_space(H5SC_t *cache, size_t bytes_needed, bool allow_oversized_single_chunk);

static inline bool H5SC__active_reclaim_required(const H5SC_t *cache, size_t bytes_needed, bool oversized);

static size_t H5SC__calc_inc_for_chunk(const H5D_t *dset, const H5SC_chunk_t *chk, size_t desired);

static herr_t H5SC__invoke_read_dset_batched(H5SC_t *cache, H5D_dset_io_info_t *dset_info,
                                             H5SC_dset_header_t *dset_hdr);
static herr_t H5SC__invoke_write_dset_batched(H5SC_t *cache, H5D_dset_io_info_t *dset_info,
                                              H5SC_dset_header_t *dset_hdr);

/* Dataset specific helper functions */

static inline H5SC_reclaim_contrib_t H5SC__chunk_reclaim_contribution(const H5SC_dset_header_t *dset_hdr,
                                                                      const H5SC_chunk_t       *chunk);

static inline herr_t H5SC__apply_reclaim_transition(H5SC_t *cache, H5SC_dset_header_t *dset_hdr,
                                                    H5SC_reclaim_contrib_t before,
                                                    H5SC_reclaim_contrib_t after);

static inline herr_t H5SC__chunk_set_dirty(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk,
                                           bool dirty);

static bool H5SC__candidate_is_reclaimable(const H5SC_dset_header_t *dset_hdr, const H5SC_chunk_t *chunk,
                                           H5SC_reclaim_policy_t policy);

static bool H5SC__select_clean_candidate(H5SC_dset_header_t *dset_hdr, H5SC_reclaim_policy_t policy,
                                         H5SC_chunk_t **candidate);

static bool H5SC__select_dirty_candidate(H5SC_dset_header_t *dset_hdr, H5SC_reclaim_policy_t policy,
                                         H5SC_chunk_t **candidate);

static bool H5SC__space_exceeded(size_t current, size_t additional, size_t limit);

static void H5SC__saturating_add_size(size_t *total, size_t amount);

static inline size_t H5SC__cached_active_reclaimable(const H5SC_t *cache);
static inline size_t H5SC__effective_budget(const H5SC_t *cache);

static inline herr_t H5SC__chunk_pin(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk);
static inline herr_t H5SC__chunk_unpin(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk);

#if H5SC_DO_SANITY_CHECKS
static size_t H5SC__calc_reclaimable_bytes(const H5SC_t *cache, H5SC_reclaim_policy_t policy);

static herr_t H5SC__verify_cached_reclaimability(const H5SC_t *cache);

static herr_t H5SC__verify_clean_candidate(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *candidate,
                                           H5SC_reclaim_policy_t policy);

static herr_t H5SC__verify_dirty_candidate(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *candidate,
                                           H5SC_reclaim_policy_t policy);
#endif

/* Stat-specific functions */
static void H5SC__stats_reset(H5SC_t *cache);
static void H5SC__stats_record_lookup(H5SC_t *cache, hbool_t hit);
static void H5SC__stats_record_insert(H5SC_t *cache);
static void H5SC__stats_record_delete(H5SC_t *cache);
static void H5SC__stats_record_chunk_flush(H5SC_t *cache);
static void H5SC__stats_record_dset_flush(H5SC_t *cache);
static void H5SC__stats_record_batch_read(H5SC_t *cache);
static void H5SC__stats_record_batch_write(H5SC_t *cache);
static void H5SC__stats_record_batch_read_len(H5SC_t *cache, size_t batch_len);
static void H5SC__stats_record_batch_write_len(H5SC_t *cache, size_t batch_len);
static void H5SC__stats_record_eviction(H5SC_t *cache);

static herr_t H5SC__estimate_resident_size(const H5SC_dset_header_t *dset_hdr, const H5D_t *dset,
                                           const H5SC_io_sel_chunk_t *sel, bool is_write, size_t write_bytes,
                                           H5SC_size_est_source_t *source, size_t *estimate);

static void H5SC__report_oversized_admission(H5SC_t *cache, size_t estimated_bytes);

#if defined(H5SC_ENABLE_STAT_DUMPS) && (H5SC_ENABLE_STAT_DUMPS + 0)

static double H5SC__stats_get_hit_rate(const H5SC_t *cache);
static double H5SC__stats_get_miss_rate(const H5SC_t *cache);
static herr_t H5SC__stats_dump(const H5SC_t *cache);
#endif

#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
static void H5SC__stats_record_size_estimate(H5SC_t *cache, H5SC_size_est_source_t source, size_t estimate,
                                             size_t actual);
#endif

/*****************************/
/* Library Private Variables */
/*****************************/

/*********************/
/* Package Variables */
/*********************/

/* Package initialization variable */
bool H5_PKG_INIT_VAR = false;

/*******************/
/* Local Variables */
/*******************/

/*******************/
/* UTHASH Funcs    */
/*******************/

/* Helpers: generic wrappers that accept the handle name tokens */
#define H5SC_HASH_FIND(hhname_, head_, keyptr_, keylen_, out_)                                               \
    HASH_FIND(hhname_, (head_), (keyptr_), (keylen_), (out_))

#define H5SC_HASH_ADD_KEYPTR(hhname_, head_, keyptr_, keylen_, item_)                                        \
    HASH_ADD_KEYPTR(hhname_, (head_), (keyptr_), (keylen_), (item_))

#define H5SC_HASH_DEL(head_, item_) HASH_DEL((head_), (item_))

#define H5SC_HASH_COUNT(head_) HASH_COUNT((head_))

/*-------------------------------------------------------------------------
 * Function: H5SC_create
 *
 * Purpose:  Creates a new, empty shared chunk cache.
 * Currently called by H5F__new when a new file object and
 * initialized. When available, default field values will be
 * overwritten by those available in the FAPL.
 *
 * Return:   Pointer to newly created cache on success, NULL on failure
 *-------------------------------------------------------------------------
 */
H5SC_t *
H5SC_create(H5F_t *file, H5P_genplist_t *fa_plist, H5SC__cache_config_t *config_ptr)
{
    H5SC_t *cache     = NULL;
    H5SC_t *ret_value = NULL;

    FUNC_ENTER_NOAPI(NULL)

    assert(file);
    assert(fa_plist);
    assert(config_ptr);
    /*
    #if H5SC_DO_SANITY_CHECKS

        fprintf(stdout, "\n\nH5SC_create(): config_ptr->version = %d.\n", config_ptr->version);
        fprintf(stdout, "               config_ptr->max_q_size = 0x%zu.\n\n",
        (size_t)(config_ptr->max_q_size)); fprintf(stdout, "               config_ptr->max_a_size =
        0x%zu.\n", (size_t)(config_ptr->max_a_size));
    #endif
    */

    /* Allocated cache struct */
    if (NULL == (cache = H5MM_calloc(sizeof(H5SC_t)))) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTALLOC, NULL, "unable to allocate buffer for shared chunk cache");
    }

    /*
     * Initialize cache fields to default values. Some values are available through FAPL configuration;
     * SCC_quiescent_limit - Set by FAPL; if zero, it is set to be 1 GB by default
     * SCC_active_limit - Set by FAPL; if zero, it is set to be 2 GB by default
     */
    cache->SCC_magic                 = H5SC_MAIN_MAGIC;
    cache->SCC_quiescent_size        = (size_t)0;
    cache->SCC_quiescent_limit       = config_ptr->max_q_size;
    cache->SCC_active_size           = (size_t)0;
    cache->SCC_active_limit          = config_ptr->max_a_size;
    cache->dset_lru_len              = (size_t)0;
    cache->reclaimable_clean_bytes   = (size_t)0;
    cache->reclaimable_dirty_bytes   = (size_t)0;
    cache->dset_lru_head_ptr         = NULL;
    cache->dset_lru_tail_ptr         = NULL;
    cache->chunk_hash_table_head_ptr = NULL;
    cache->dset_hash_table_head_ptr  = NULL;

#if H5SC_DO_SANITY_CHECKS
    cache->test_fail_next_chunk_flush = false;
#endif

    /* Success */
    H5SC__stats_reset(cache);
    ret_value = cache;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_create() */

/*-------------------------------------------------------------------------
 * Function: H5SC_destroy
 *
 * Purpose:
 *   Destroy an already-empty shared chunk cache and release its allocation.
 *
 *   This function does not flush or evict dataset chunks. Dataset close or
 *   higher-level cache teardown must first flush-and-evict all dataset state.
 *   Destruction fails if dataset headers, chunk entries, or accounted resident
 *   bytes remain.
 *
 * Return:   SUCCEED on success, FAIL on failure
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_destroy(H5SC_t *cache)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

#if defined(H5SC_ENABLE_STAT_DUMPS) && (H5SC_ENABLE_STAT_DUMPS + 0)
    {
        bool have_stats = (cache->stats.scc_lookups != 0);

#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
        have_stats = have_stats || (cache->stats.size_estimate_count != 0);
#endif

        /* Dump stats before they are reset. */
        if (have_stats) {
            fprintf(stdout, "\n=== SCC Stats Prior to Cache Destruction ===\n");

            if (H5SC__stats_dump(cache) < 0)
                printf("unable to dump SCC statistics\n");
        }
    }
#endif

    H5SC__stats_reset(cache);

    if (cache->dset_lru_len == 0 && cache->SCC_quiescent_size == 0 && cache->reclaimable_clean_bytes == 0 &&
        cache->reclaimable_dirty_bytes == 0) {

        H5SC__reset_hash_tables(cache);

        H5MM_free(cache);
        cache = NULL;
    }
    else {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTFREE, FAIL, "failed to free cache contents prior to destruction");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_destroy() */

/*-------------------------------------------------------------------------
 * Function: H5SC_flush
 *
 * Purpose:  Writes all dirty chunks in a shared chunk cache to disk while
 * retaining the chunks and dataset headers in the cache.
 *
 * Return:   SUCCEED on success, FAIL on failure
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_flush(H5SC_t *cache)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    for (H5SC_dset_header_t *dset_hdr = cache->dset_lru_tail_ptr; dset_hdr;
         dset_hdr                     = dset_hdr->prev_dset_ptr) {
        if (!dset_hdr->dset)
            HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "unable to flush SCC dataset without dataset state");

        if (H5SC_flush_dset(cache, dset_hdr->dset, false) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTFLUSH, FAIL, "unable to flush SCC dataset");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_flush() */

/*-------------------------------------------------------------------------
 * Function: H5SC__io_scratch_free_storage
 *
 * Purpose:
 *   Release all backing allocations owned directly by an
 *     H5SC_io_scratch_t structure without freeing the structure itself.
 *
 *   Pointer-valued entries stored within the scratch vectors are treated as
 *     non-owning aliases. In particular, this function does not release
 *     layout-specific udata objects, resident chunk objects, encoded write
 *     or H5SC_chunk_t objects referenced by chunk[].
 *
 *   After all owned backing allocations are released, the structure is
 *     reset to an empty state with alloced == 0 and all pointer fields NULL.
 *
 * Inputs:
 *   H5SC_io_scratch_t *scratch:
 *     Scratch structure whose directly owned backing storage is to be
 *     released. May be NULL.
 *
 * Return:
 *   None.
 *-------------------------------------------------------------------------
 */
static void
H5SC__io_scratch_free_storage(H5SC_io_scratch_t *scratch)
{
    if (!scratch)
        return;

    scratch->scaled = (const hsize_t **)H5MM_xfree((void *)scratch->scaled);

    scratch->addr                     = (haddr_t **)H5MM_xfree(scratch->addr);
    scratch->size                     = (hsize_t **)H5MM_xfree(scratch->size);
    scratch->defined_values_size      = (hsize_t **)H5MM_xfree(scratch->defined_values_size);
    scratch->size_hint                = (size_t **)H5MM_xfree(scratch->size_hint);
    scratch->defined_values_size_hint = (size_t **)H5MM_xfree(scratch->defined_values_size_hint);
    scratch->udata                    = (void **)H5MM_xfree(scratch->udata);

    scratch->addr_values                     = (haddr_t *)H5MM_xfree(scratch->addr_values);
    scratch->size_values                     = (hsize_t *)H5MM_xfree(scratch->size_values);
    scratch->defined_values_size_values      = (hsize_t *)H5MM_xfree(scratch->defined_values_size_values);
    scratch->size_hint_values                = (size_t *)H5MM_xfree(scratch->size_hint_values);
    scratch->defined_values_size_hint_values = (size_t *)H5MM_xfree(scratch->defined_values_size_hint_values);

    scratch->miss_idx = (size_t *)H5MM_xfree(scratch->miss_idx);
    scratch->hit_idx  = (size_t *)H5MM_xfree(scratch->hit_idx);
    scratch->chunk    = (void **)H5MM_xfree(scratch->chunk);

    scratch->lookup_scaled              = (const hsize_t **)H5MM_xfree((void *)scratch->lookup_scaled);
    scratch->lookup_addr                = (haddr_t **)H5MM_xfree(scratch->lookup_addr);
    scratch->lookup_size                = (hsize_t **)H5MM_xfree(scratch->lookup_size);
    scratch->lookup_defined_values_size = (hsize_t **)H5MM_xfree(scratch->lookup_defined_values_size);
    scratch->lookup_size_hint           = (size_t **)H5MM_xfree(scratch->lookup_size_hint);
    scratch->lookup_defined_values_size_hint =
        (size_t **)H5MM_xfree(scratch->lookup_defined_values_size_hint);
    scratch->lookup_udata = (void **)H5MM_xfree(scratch->lookup_udata);
    scratch->lookup_idx   = (size_t *)H5MM_xfree(scratch->lookup_idx);
    scratch->order_idx    = (size_t *)H5MM_xfree(scratch->order_idx);

    scratch->alloced = 0;
} /* end H5SC__io_scratch_free_storage */

/*-------------------------------------------------------------------------
 * Function: H5SC__io_scratch_ensure
 *
 * Purpose:
 *   Ensure that the reusable scratch storage associated with an
 *     H5SC_io_info_t structure can represent at least "needed" chunk
 *     entries.
 *
 *   Scratch storage is allocated lazily. When additional capacity is
 *     required, this function allocates a complete replacement set of
 *     callback pointer vectors, contiguous callback scalar backing arrays,
 *     and general per-chunk read/write working vectors. The replacement is
 *     committed only after every allocation succeeds, so an allocation
 *     failure leaves any previously existing scratch storage unchanged.
 *
 *   The callback-facing pointer vectors are bound to their corresponding
 *     contiguous backing arrays before the new storage is committed:
 *
 *       addr[i]                     -> &addr_values[i]
 *       size[i]                     -> &size_values[i]
 *       defined_values_size[i]      -> &defined_values_size_values[i]
 *       size_hint[i]                -> &size_hint_values[i]
 *       defined_values_size_hint[i] -> &defined_values_size_hint_values[i]
 *
 *   This function manages capacity only. Callers must initialize the values
 *     they intend to use before invoking a layout callback.
 *
 *   Additional scratch vectors provide reusable per-chunk storage for
 *     scaled-coordinate references, layout udata references, miss/hit
 *     indices and resident/decoded chunk pointers. These vectors own only their backing allocations;
 *     pointer-valued entries do not imply ownership of the objects to which
 *     they refer.
 *
 *   The lookup_* vectors provide a reusable compact callback workspace.
 *     They are populated only over the prefix required by the current
 *     callback invocation and may alias fields in H5SC_io_sel_chunk_t or
 *     entries in the primary per-chunk scratch vectors. The same workspace
 *     may be reused sequentially for cache-miss lookup and resident-hit
 *     metadata refresh; its contents have no validity between callback
 *     invocations.
 *
 * Inputs:
 *   H5SC_io_info_t *sc_io_info:
 *     Dataset-owned SCC I/O state whose scratch storage is to be allocated
 *     or enlarged.
 *
 *   size_t needed:
 *     Minimum number of scratch entries required by the caller.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__io_scratch_ensure(H5SC_io_info_t *sc_io_info, size_t needed)
{
    H5SC_io_scratch_t *scratch = NULL;

    H5SC_io_scratch_t new_scratch = {0};

    size_t new_alloc;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(sc_io_info);

    if (needed == 0)
        HGOTO_DONE(SUCCEED);

    /*
     * Allocate the scratch descriptor lazily. H5MM_calloc() is intentional:
     * scratch fields not yet converted to reusable storage remain NULL.
     */
    if (!sc_io_info->scratch) {
        if (NULL == (sc_io_info->scratch = H5MM_calloc(sizeof(H5SC_io_scratch_t))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC I/O scratch structure");
    }

    scratch = sc_io_info->scratch;

    if (scratch->alloced >= needed)
        HGOTO_DONE(SUCCEED);

    /*
     * Grow geometrically to reduce the number of future allocation events.
     * Begin with the same baseline currently used by sel_chunks.
     */
    new_alloc = scratch->alloced ? scratch->alloced : H5SC_INIT_CHUNK_LIST_SIZE;

    while (new_alloc < needed) {
        if (new_alloc > (SIZE_MAX / 2)) {
            new_alloc = needed;
            break;
        }

        new_alloc *= 2;
    }

    if (new_alloc < needed)
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "unable to represent requested SCC scratch capacity");

    /*
     * Allocate a complete replacement set. Do not modify scratch until every
     * allocation succeeds.
     */
    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.addr)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "SCC scratch address-vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "SCC scratch size-vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.defined_values_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch defined-values size-vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.size_hint)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch size-hint vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.defined_values_size_hint)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch defined-values size-hint vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.addr_values)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch address backing-storage allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.size_values)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch size backing-storage allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.defined_values_size_values)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch defined-values backing-storage allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.size_hint_values)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch size-hint backing-storage allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.defined_values_size_hint_values)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch defined-values size-hint backing-storage allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.scaled)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "SCC scratch scaled-vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.udata)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "SCC scratch udata-vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.miss_idx)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "SCC scratch miss-index allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.hit_idx)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "SCC scratch hit-index allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.chunk)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL, "SCC scratch chunk-vector allocation size overflow");

    if (NULL == (new_scratch.addr = H5MM_malloc(new_alloc * sizeof(*new_scratch.addr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch address vector");

    if (NULL == (new_scratch.size = H5MM_malloc(new_alloc * sizeof(*new_scratch.size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch size vector");

    if (NULL ==
        (new_scratch.defined_values_size = H5MM_malloc(new_alloc * sizeof(*new_scratch.defined_values_size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch defined-values size vector");

    if (NULL == (new_scratch.size_hint = H5MM_malloc(new_alloc * sizeof(*new_scratch.size_hint))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch size-hint vector");

    if (NULL == (new_scratch.defined_values_size_hint =
                     H5MM_malloc(new_alloc * sizeof(*new_scratch.defined_values_size_hint))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch defined-values size-hint vector");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.lookup_scaled)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch lookup-scaled vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.lookup_addr)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch lookup-address vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.lookup_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch lookup-size vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.lookup_defined_values_size)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch lookup-defined-values-size vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.lookup_size_hint)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch lookup-size-hint vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.lookup_defined_values_size_hint)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch lookup-defined-values-size-hint vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.lookup_udata)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch lookup-udata vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.lookup_idx)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch lookup-index vector allocation size overflow");

    if (new_alloc > (SIZE_MAX / sizeof(*new_scratch.order_idx)))
        HGOTO_ERROR(H5E_RESOURCE, H5E_OVERFLOW, FAIL,
                    "SCC scratch order-index vector allocation size overflow");

    /*
     * Backing arrays are zero-initialized because zero is the normal default
     * for all scalar outputs except the chunk address.
     */

    if (NULL == (new_scratch.addr_values = H5MM_malloc(new_alloc * sizeof(*new_scratch.addr_values))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch address backing storage");

    if (NULL == (new_scratch.size_values = H5MM_calloc(new_alloc * sizeof(*new_scratch.size_values))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch size backing storage");

    if (NULL == (new_scratch.defined_values_size_values =
                     H5MM_calloc(new_alloc * sizeof(*new_scratch.defined_values_size_values))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch defined-values size backing storage");

    if (NULL ==
        (new_scratch.size_hint_values = H5MM_calloc(new_alloc * sizeof(*new_scratch.size_hint_values))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch size-hint backing storage");

    if (NULL == (new_scratch.defined_values_size_hint_values =
                     H5MM_calloc(new_alloc * sizeof(*new_scratch.defined_values_size_hint_values))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch defined-values size-hint backing storage");

    if (NULL == (new_scratch.scaled = H5MM_malloc(new_alloc * sizeof(*new_scratch.scaled))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch scaled vector");

    if (NULL == (new_scratch.udata = H5MM_calloc(new_alloc * sizeof(*new_scratch.udata))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch udata vector");

    if (NULL == (new_scratch.miss_idx = H5MM_malloc(new_alloc * sizeof(*new_scratch.miss_idx))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch miss-index vector");

    if (NULL == (new_scratch.hit_idx = H5MM_malloc(new_alloc * sizeof(*new_scratch.hit_idx))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch hit-index vector");

    if (NULL == (new_scratch.chunk = H5MM_calloc(new_alloc * sizeof(*new_scratch.chunk))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch chunk vector");

    if (NULL == (new_scratch.lookup_scaled = H5MM_malloc(new_alloc * sizeof(*new_scratch.lookup_scaled))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch lookup-scaled vector");

    if (NULL == (new_scratch.lookup_addr = H5MM_malloc(new_alloc * sizeof(*new_scratch.lookup_addr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch lookup-address vector");

    if (NULL == (new_scratch.lookup_size = H5MM_malloc(new_alloc * sizeof(*new_scratch.lookup_size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch lookup-size vector");

    if (NULL == (new_scratch.lookup_defined_values_size =
                     H5MM_malloc(new_alloc * sizeof(*new_scratch.lookup_defined_values_size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch lookup-defined-values-size vector");

    if (NULL ==
        (new_scratch.lookup_size_hint = H5MM_malloc(new_alloc * sizeof(*new_scratch.lookup_size_hint))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch lookup-size-hint vector");

    if (NULL == (new_scratch.lookup_defined_values_size_hint =
                     H5MM_malloc(new_alloc * sizeof(*new_scratch.lookup_defined_values_size_hint))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                    "unable to allocate SCC scratch lookup-defined-values-size-hint vector");

    if (NULL == (new_scratch.lookup_udata = H5MM_calloc(new_alloc * sizeof(*new_scratch.lookup_udata))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch lookup-udata vector");

    if (NULL == (new_scratch.lookup_idx = H5MM_malloc(new_alloc * sizeof(*new_scratch.lookup_idx))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch lookup-index vector");

    if (NULL == (new_scratch.order_idx = H5MM_malloc(new_alloc * sizeof(*new_scratch.order_idx))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate SCC scratch order-index vector");

    /* miss_idx and hit_idx do not need initialization; every valid entry is explicitly assigned before
     * being read.
     */

    /*
     * Bind the callback-facing vector entries to the contiguous scalar
     * backing arrays.
     */

    for (size_t i = 0; i < new_alloc; i++) {
        new_scratch.addr_values[i] = HADDR_UNDEF;

        new_scratch.addr[i] = &new_scratch.addr_values[i];

        new_scratch.size[i] = &new_scratch.size_values[i];

        new_scratch.defined_values_size[i] = &new_scratch.defined_values_size_values[i];

        new_scratch.size_hint[i] = &new_scratch.size_hint_values[i];

        new_scratch.defined_values_size_hint[i] = &new_scratch.defined_values_size_hint_values[i];
    }

    /*
     * Record the capacity represented by the replacement storage.
     */
    new_scratch.alloced = new_alloc;

    /*
     * All replacement allocations succeeded. The old scratch values are
     * transient and do not need to be copied.
     */
    H5SC__io_scratch_free_storage(scratch);

    *scratch = new_scratch;

#if H5SC_DO_SANITY_CHECKS
    assert(scratch->alloced == new_alloc);
#endif

    /*
     * Ownership has moved into *scratch. Clear the local descriptor so the
     * done: cleanup cannot release the committed storage.
     */
    memset(&new_scratch, 0, sizeof(new_scratch));

done:
    /*
     * These pointers are non-NULL only if replacement construction failed
     * before ownership was transferred.
     */

    H5SC__io_scratch_free_storage(&new_scratch);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__io_scratch_ensure() */

/*-------------------------------------------------------------------------
 * Function: H5SC__get_defined_chunk
 *
 * Purpose:
 *   Determine which elements selected from one logical structured chunk are
 *     currently defined.
 *
 *   Resident SCC state is authoritative. If cached_chk contains a decoded
 *     chunk object, the layout client's defined_values callback is applied
 *     directly to that resident object so that dirty or otherwise unflushed
 *     in-memory defined-value state is reflected in the result.
 *
 *   If no resident decoded object is available, the function performs a
 *     cache-neutral layout lookup:
 *
 *       - an undefined on-disk chunk address means that no selected values
 *         from the chunk are defined;
 *
 *       - an allocated chunk with defined_values_size == 0 is treated as
 *         fully defined, allowing sel->file_space to be returned without an
 *         on-disk metadata read;
 *
 *       - otherwise, only the encoded defined-value metadata is read and
 *         decoded.
 *
 *   Metadata-only decoded chunk objects are temporary. They are never added
 *     to SCC and are released through the layout client's evict callback.
 *     The returned dataspace is explicitly copied before that temporary
 *     object is released so its lifetime is independent of the decoded
 *     layout representation.
 *
 *   This function does not create H5SC_chunk_t entries, insert chunks into
 *     SCC, modify either LRU, pin chunks, or alter SCC cache accounting.
 *
 * Inputs:
 *   H5D_t *dset:
 *     Pointer to the dataset whose structured-chunk layout callbacks and
 *     file storage are queried.
 *
 *   const H5SC_io_sel_chunk_t *sel:
 *     Selected-chunk information. sel->file_space contains the query
 *     selection in logical chunk coordinates and sel->scaled identifies
 *     the chunk within the dataset.
 *
 *   H5SC_chunk_t *cached_chk:
 *     Existing SCC chunk entry, if one is present. May be NULL. A non-NULL
 *     entry with a resident chunk_obj is queried directly and takes
 *     precedence over the on-disk representation.
 *
 *   bool filtered:
 *     True when the dataset has a structured-chunk filter pipeline.
 *
 *   bool partial_bound_chunks_different_encoding:
 *     True when partial-edge chunks use encoding behavior different from
 *     complete chunks. Used when determining how metadata-only disk state
 *     must be decoded.
 *
 * Outputs:
 *   H5S_t **chunk_defined:
 *     On success, receives a newly allocated dataspace containing the
 *     intersection of sel->file_space and the currently defined values in
 *     the logical chunk.
 *
 *     The output may be NULL on successful return when neither a resident
 *     nor an allocated on-disk representation exists for the chunk. A
 *     non-NULL returned dataspace is owned by the caller and must be closed
 *     with H5S_close().
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__get_defined_chunk(H5D_t *dset, const H5SC_io_sel_chunk_t *sel, H5SC_chunk_t *cached_chk, bool filtered,
                        bool partial_bound_chunks_different_encoding, H5S_t **chunk_defined)
{
    const hsize_t *scaled_arr[1];
    haddr_t       *addr_arr[1];
    hsize_t       *size_arr[1];
    hsize_t       *def_size_arr[1];
    size_t        *size_hint_arr[1];
    size_t        *def_hint_arr[1];
    void          *udata_arr[1] = {NULL};

    haddr_t addr_local      = HADDR_UNDEF;
    hsize_t size_local      = 0;
    hsize_t def_size_local  = 0;
    size_t  size_hint_local = 0;
    size_t  def_hint_local  = 0;

    void  *tmp_chunk   = NULL;
    bool   tmp_decoded = false;
    herr_t ret_value   = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(dset->shared);
    assert(dset->shared->layout.sc_ops);
    assert(sel);
    assert(sel->file_space);
    assert(chunk_defined);

    *chunk_defined = NULL;

    /*
     * Resident SCC state is authoritative. This is particularly important
     * for dirty chunks whose current defined-value selection may not yet
     * match the on-disk representation.
     */
    if (cached_chk && cached_chk->chunk_obj) {
        if (dset->shared->layout.sc_ops->defined_values) {
            if (dset->shared->layout.sc_ops->defined_values(dset, sel->file_space, cached_chk->chunk_obj,
                                                            chunk_defined, cached_chk->udata) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query resident chunk defined values");
        }
        else {
            /*
             * A missing defined_values callback means that all values in the
             * resident chunk are defined.
             */
            if (NULL == (*chunk_defined = H5S_copy(sel->file_space, false, true)))
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy resident chunk selection");
        }

        HGOTO_DONE(SUCCEED);
    }

    /*
     * The remaining paths require an on-disk lookup, but do not create or
     * attach an SCC entry.
     */
    if (!dset->shared->layout.sc_ops->lookup)
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "layout does not provide a chunk lookup callback");

    scaled_arr[0] = sel->scaled;

    addr_arr[0]      = &addr_local;
    size_arr[0]      = &size_local;
    def_size_arr[0]  = &def_size_local;
    size_hint_arr[0] = &size_hint_local;
    def_hint_arr[0]  = &def_hint_local;

    if (dset->shared->layout.sc_ops->lookup(dset, 1, scaled_arr, addr_arr, size_arr, def_size_arr,
                                            size_hint_arr, def_hint_arr, udata_arr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to look up chunk for defined-value query");

    /*
     * Sparse miss: no on-disk chunk exists and there is no resident decoded
     * chunk, so nothing from this logical chunk is defined.
     */
    if (!H5_addr_defined(addr_local))
        HGOTO_DONE(SUCCEED);

    /*
     * Fast path: for an allocated chunk, defined_values_size == 0 means all
     * values are defined. The selected portion of the logical chunk can be
     * returned directly without reading or decoding metadata.
     */
    if (def_size_local == 0) {
        if (NULL == (*chunk_defined = H5S_copy(sel->file_space, false, true)))
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy fully defined chunk selection");

        HGOTO_DONE(SUCCEED);
    }

    /*
     * A nonzero defined-value metadata section must be read and decoded.
     */
    if (!dset->shared->layout.sc_ops->decode_defined_values)
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "layout does not provide defined-value decoding");

    if (!dset->shared->layout.sc_ops->defined_values)
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL,
                    "layout does not provide a defined-values query callback");

    if (!dset->shared->layout.sc_ops->evict)
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "layout does not provide temporary chunk cleanup");

    {
        bool   partial_bound = false;
        size_t read_size;
        size_t decode_nbytes;
        size_t decode_alloc;

        if (filtered && partial_bound_chunks_different_encoding &&
            H5D__chunk_is_partial_edge_chunk(dset->shared->ndims, dset->shared->layout.u.struct_chunk.dim,
                                             sel->scaled, dset->shared->curr_dims))
            partial_bound = true;

        H5_CHECK_OVERFLOW(def_size_local, hsize_t, size_t);
        read_size = (size_t)def_size_local;

        decode_nbytes = read_size;
        decode_alloc  = MAX(def_hint_local, read_size);

        if (decode_alloc == 0)
            HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "invalid defined-value metadata allocation size");

        if (NULL == (tmp_chunk = H5MM_malloc(decode_alloc)))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate defined-value metadata buffer");

        if (H5F_block_read(dset->oloc.file, H5FD_MEM_DRAW, addr_local, read_size, tmp_chunk) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "unable to read defined-value metadata");

        /*
         * On failure, the hardened decode_defined_values callback leaves
         * tmp_chunk referring to the original raw buffer. On success it
         * replaces tmp_chunk with the layout-specific decoded object.
         */
        if (dset->shared->layout.sc_ops->decode_defined_values(dset, &decode_nbytes, &decode_alloc,
                                                               partial_bound, &tmp_chunk, udata_arr[0]) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTDECODE, FAIL, "unable to decode defined-value metadata");

        tmp_decoded = true;

        {
            H5S_t *tmp_defined = NULL;

            /*
             * Query the layout-specific temporary decoded representation.
             *
             * Do not expose this returned dataspace beyond the lifetime of
             * tmp_chunk. Depending on the internal selection representation,
             * selection state returned by the callback may retain references to
             * the decoded chunk's sel_space.
             */
            if (dset->shared->layout.sc_ops->defined_values(dset, sel->file_space, tmp_chunk, &tmp_defined,
                                                            udata_arr[0]) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query decoded defined values");

            assert(tmp_defined);

            /*
             * Create a fully detached dataspace that remains valid after the
             * metadata-only decoded chunk is evicted below.
             */
            if (NULL == (*chunk_defined = H5S_copy(tmp_defined, false, true))) {
                if (H5S_close(tmp_defined) < 0)
                    HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                                "unable to close temporary defined-value selection");

                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to detach defined-value selection");
            }

            if (H5S_close(tmp_defined) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                            "unable to close temporary defined-value selection");
        }
    }

done:
    /*
     * Release temporary disk-query state. A successfully decoded
     * metadata-only chunk is owned by the layout implementation and must be
     * released through its evict callback.
     */
    if (tmp_chunk) {
        if (tmp_decoded) {
            herr_t evict_ret;

            evict_ret = dset->shared->layout.sc_ops->evict(dset, tmp_chunk, udata_arr[0]);

            /*
             * Ownership has been handed to the callback regardless of its
             * return value. Do not attempt generic cleanup afterward.
             */
            tmp_chunk    = NULL;
            udata_arr[0] = NULL;

            if (evict_ret < 0)
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "unable to release temporary decoded chunk");
        }
        else
            tmp_chunk = H5MM_xfree(tmp_chunk);
    }

    /*
     * If no decoded object consumed udata, it remains caller-owned lookup
     * scratch and can be released directly.
     */
    if (udata_arr[0])
        udata_arr[0] = H5MM_xfree(udata_arr[0]);

    if (ret_value < 0 && *chunk_defined) {
        if (H5S_close(*chunk_defined) < 0)
            HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                        "unable to release defined-value selection after error");

        *chunk_defined = NULL;
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__get_defined_chunk() */

/*-------------------------------------------------------------------------
 * Function: H5SC__chunk_teardown_resident
 *
 * Purpose:
 *   Function that tears down the resident, decoded chunk state associated
 *     with an SCC chunk entry. If a decoded chunk object is present, the
 *     function attempts to release it using the dataset layout client's
 *     eviction callback and clears the associated SCC pointers and cached
 *     size metadata.
 *
 *   For layouts that provide an sc_ops->evict callback, that callback is
 *     responsible for releasing the decoded chunk object and any associated
 *     udata. If no callback is available, this function falls back to
 *     freeing SCC-owned resident pointers directly as a best-effort leak
 *     prevention measure.
 *
 * Inputs:
 *   H5D_t *dset:
 *     Pointer to the dataset whose layout callbacks define how resident
 *     chunk state is to be released.
 *
 *   H5SC_chunk_t *chunk:
 *     Pointer to the SCC chunk entry whose resident decoded state is to be
 *     torn down.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__chunk_teardown_resident(H5D_t *dset, H5SC_chunk_t *chunk)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(dset->shared);
    assert(dset->shared->layout.sc_ops);
    assert(chunk);

    if (chunk->chunk_obj) {
        if (dset->shared->layout.sc_ops->evict) {
            if (dset->shared->layout.sc_ops->evict(dset, chunk->chunk_obj, chunk->udata) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTRELEASE, FAIL, "unable to evict resident chunk object");

            /* Evict callback releases the resident payloads. */
            chunk->chunk_obj = NULL;
            chunk->udata     = NULL;
        }
        else {
            /* Fallback path when no layout-specific evict callback exists. */
            chunk->chunk_obj = H5MM_xfree(chunk->chunk_obj);
            chunk->udata     = H5MM_xfree(chunk->udata);
        }

        /*
         * The caller must remove the chunk from its dataset LRU and reconcile cache
         * accounting before invoking this helper. Resident teardown therefore does
         * not independently update dataset size or reclaimability counters.
         */
        chunk->cached_chunk_size = 0;
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__chunk_teardown_resident() */

/*-------------------------------------------------------------------------
 * Function: H5SC__flush_one_chunk
 *
 * Purpose:
 *   Function that flushes a single dirty, resident SCC chunk to disk. The
 *     function performs any required layout lookup, encodes the resident
 *     chunk object into an on-disk buffer, updates or allocates the chunk’s
 *     on-disk location through the layout client's insert callback, writes
 *     the encoded buffer to the file, and then updates the SCC chunk state
 *     to reflect the new on-disk metadata.
 *
 *   The function recomputes partial-boundary encoding behavior using the
 *     same layout policy queried by the write path so that flush-time
 *     encoding remains consistent with chunk writes. Shell chunks that are
 *     neither resident nor defined on disk are skipped. Dirty chunks are
 *     required to have a resident decoded chunk object.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state used for statistics and cache-related
 *     bookkeeping.
 *
 *   H5D_t *dset:
 *     Pointer to the dataset whose layout callbacks and file handle are used
 *     for lookup, encoding, insertion, and write-back.
 *
 *   H5SC_dset_header_t *dset_hdr:
 *     Pointer to the SCC dataset header associated with the chunk being
 *     flushed.
 *
 *   H5SC_chunk_t *chk:
 *     Pointer to the SCC chunk entry to flush. The chunk must be valid, and
 *     if marked dirty, must have a resident decoded chunk object.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__flush_one_chunk(H5SC_t *cache, H5D_t *dset, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chk)
{
    herr_t ret_value = SUCCEED;

    /* Callback scratch */
    const hsize_t *scaled_arr[1];
    haddr_t       *addr_arr[1];
    hsize_t       *size_arr[1];
    hsize_t       *def_sz_arr[1];
    size_t        *size_hint_arr[1];
    size_t        *def_sz_hint_arr[1];
    void          *udata_arr[1] = {NULL};
    haddr_t        addr_local;
    hsize_t        size_local;
    hsize_t        def_local;
    size_t         hint_local;
    size_t         def_hint_local;

    haddr_t md_tag     = HADDR_UNDEF;
    bool    md_tag_set = false;

    /* Encode/insert scratch */
    hsize_t          old_disk_size[1];
    void            *write_buf                               = NULL;
    hsize_t          write_size                              = 0;
    hsize_t          write_alloc                             = 0;
    bool             partial_bound                           = false;
    bool             partial_bound_chunks_different_encoding = false;
    H5O_stc_pline_t *pline                                   = NULL;
    hbool_t          filtered                                = false;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset_hdr);
    assert(chk);
    assert(dset->shared->layout.sc_ops);
    assert(dset->shared->layout.sc_ops->lookup);
    assert(dset->shared->layout.sc_ops->encode);
    assert(dset->shared->layout.sc_ops->insert);

    /*
     * This helper persists dirty resident state only. Clean chunks and empty
     * shell entries require no layout lookup, encoding, insertion, or I/O.
     */
    if (!chk->dirty_flag)
        HGOTO_DONE(SUCCEED);

    if (!chk->chunk_obj)
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "flush: dirty chunk has no resident object");

#if H5SC_DO_SANITY_CHECKS
    if (cache->test_fail_next_chunk_flush)
        HGOTO_ERROR(H5E_SCC, H5E_WRITEERROR, FAIL, "injected dirty-chunk flush failure");
#endif

    /* Only dirty resident chunks reach metadata operations. */
    H5AC_tag(dset->oloc.addr, &md_tag);
    md_tag_set = true;

    /* Determine partial-boundary encoding behavior. */
    if (dset->shared->layout.sc_ops->layout_query(dset, NULL, NULL,
                                                  &partial_bound_chunks_different_encoding) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query partial-bound encoding policy (SCC)");

    pline = &(dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects)
        filtered = true;

    if (filtered && partial_bound_chunks_different_encoding &&
        H5D__chunk_is_partial_edge_chunk(dset->shared->ndims, dset->shared->layout.u.struct_chunk.dim,
                                         chk->scaled, dset->shared->curr_dims))
        partial_bound = true;

    /* Build single-entry lookup argument arrays. */
    scaled_arr[0] = chk->scaled;

    addr_local     = chk->disk_addr;
    size_local     = (hsize_t)chk->disk_nbytes;
    def_local      = 0;
    hint_local     = 0;
    def_hint_local = 0;

    addr_arr[0]        = &addr_local;
    size_arr[0]        = &size_local;
    def_sz_arr[0]      = &def_local;
    size_hint_arr[0]   = &hint_local;
    def_sz_hint_arr[0] = &def_hint_local;

    if (dset->shared->layout.sc_ops->lookup(dset, 1, scaled_arr, addr_arr, size_arr, def_sz_arr,
                                            size_hint_arr, def_sz_hint_arr, udata_arr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "flush: lookup callback failed (SCC)");

    old_disk_size[0] = H5_addr_defined(addr_local) ? size_local : 0;

    /* Encode the resident chunk into a temporary write buffer. */
    if (dset->shared->layout.sc_ops->encode(dset, &write_size, &write_alloc, partial_bound, chk->chunk_obj,
                                            udata_arr[0], &write_buf) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTENCODE, FAIL, "flush: encode callback failed (SCC)");

    /* Encode the resident chunk into a temporary write buffer. */
    if (dset->shared->layout.sc_ops->insert(dset, 1, scaled_arr, addr_arr, old_disk_size, &write_size, NULL,
                                            udata_arr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINSERT, FAIL, "flush: insert callback failed (SCC)");

    /* Write the encoded chunk to disk. */
    if (H5F_block_write(dset->oloc.file, H5FD_MEM_DRAW, addr_local, (size_t)write_size, write_buf) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "flush: block write failed (SCC)");

    /* Commit on-disk metadata, mark clean */
    chk->disk_addr   = addr_local;
    chk->disk_nbytes = (size_t)write_size;

    if (H5SC__chunk_set_dirty(cache, dset_hdr, chk, false) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "unable to update reclaimability after chunk flush");

    H5SC__stats_record_chunk_flush(cache);
    chk->last_op = H5SC_TAG_FLUSH_DIRTY;

#if H5SC_DO_SANITY_CHECKS
    if (H5SC__verify_cached_reclaimability(cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "cached reclaimability mismatch after chunk flush");
#endif

done:
    if (write_buf)
        write_buf = H5MM_xfree(write_buf);
    if (udata_arr[0])
        udata_arr[0] = H5MM_xfree(udata_arr[0]);

    if (md_tag_set)
        H5AC_tag(md_tag, NULL);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__flush_one_chunk() */

/*-------------------------------------------------------------------------
 * Function: H5SC__evict_one_chunk
 *
 * Purpose:
 *   Function that evicts a single SCC chunk entry from the cache. The
 *     function tears down any resident decoded chunk state, removes the
 *     chunk from the owning dataset's chunk LRU list, updates SCC cache
 *     accounting, removes the chunk from the global chunk hash table, and
 *     frees the SCC chunk structure itself.
 *
 *   This helper is intended to remove a clean, non-pinned chunk from SCC.
 *     It does not flush dirty chunks before eviction; callers are
 *     responsible for ensuring that dirty data has already been flushed or
 *     otherwise handled before invoking this function.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state containing global accounting,
 *     statistics, and the chunk hash table.
 *
 *   H5D_t *dset:
 *     Pointer to the dataset whose layout-specific resident chunk teardown
 *     behavior is required during eviction.
 *
 *   H5SC_dset_header_t *dset_hdr:
 *     Pointer to the SCC dataset header that owns the chunk being evicted.
 *     Used for dataset-local LRU removal and cache-size accounting.
 *
 *   H5SC_chunk_t *chk:
 *     Pointer to the SCC chunk entry to evict. The chunk must be valid and
 *     must not be pinned for in-flight I/O. Under sanity checks, the chunk
 *     is also expected to be clean.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__evict_one_chunk(H5SC_t *cache, H5D_t *dset, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chk)
{
    herr_t ret_value     = SUCCEED;
    size_t old_dset_size = 0;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset_hdr);
    assert(chk);
    assert(chk->magic == H5SC_CHUNK_MAGIC);

    /* Pinned chunks are in-flight for an I/O request and must not be evicted. */
    if (chk->chunk_counter > 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL, "attempt to evict a pinned/in-flight chunk");
    }

    if (chk->dirty_flag)
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL, "attempt to evict a dirty chunk");

    old_dset_size = dset_hdr->curr_dset_size;

    /* Remove the chunk from the dataset-local LRU. */
    if (H5SC__chunk_lru_remove(cache, dset_hdr, chk) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL, "failed to remove chunk from dataset LRU");
    }

    /* Update global cache accounting after unlink. */
    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "failed to update global size during chunk eviction");
    }

    /* Tear down resident decoded state. */
    if (H5SC__chunk_teardown_resident(dset, chk) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CANTRELEASE, FAIL, "failed to tear down resident chunk state");
    }

    /* Remove the chunk from the global chunk hash. */
    if (H5SC__ht_chunk_delete(cache, &chk->data_key) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL, "failed to remove chunk from hash table");
    }

    H5SC__stats_record_eviction(cache);

    chk = H5MM_xfree(chk);

#if H5SC_DO_SANITY_CHECKS
    if (H5SC__verify_cached_reclaimability(cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "cached reclaimability mismatch after chunk eviction");
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__evict_one_chunk() */

/*-------------------------------------------------------------------------
 * Function: H5SC_flush_dset
 *
 * Purpose:  Flushes all data cached for a single dataset.
 * If evict is true, also evicts all cached data.
 *
 * Inputs:
 *  H5SC_t *cache:
 *      Pointer to the main SCC struct.
 *
 *  H5D_t *dset:
 *      Pointer to the dataset that will be operated on.
 *      Assumed to be present within the SCC.
 *
 * bool evict_after_flush:
 *      Boolean used to toggle whether data will be be evicted from the cache.
 *
 *      evict_after_flush == false:
 *          Flush every dirty resident chunk and retain all entries without
 *          changing dataset-local LRU ordering.
 *
 *      evict_after_flush == true:
 *          Flush every dirty resident chunk first, then evict every resident,
 *          clean, and zero-byte shell entry. On success the dataset chunk LRU
 *          and corresponding chunk hash entries are empty.
 *
 * Return:   SUCCEED on success, FAIL on failure
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_flush_dset(H5SC_t *cache, H5D_t *dset, bool evict_after_flush)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *dset_hdr  = NULL;
    H5SC_chunk_t       *chk       = NULL;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset->shared->layout.sc_ops);

    /* Look up the dataset in the hash table to get the necessary pointer(s). */

    dset_hdr = H5SC__ht_dset_find(cache, dset->oloc.addr);
    if (!dset_hdr || !dset_hdr->lru_tail_ptr) {
        /* Dataset not found; assumed to be flushed already */
        return SUCCEED;
    }
    /* Flush the dataset
     * Scan through the LRU list for this dataset. For each chunk, if it is dirty, write the chunk to
     * disk, then mark it as clean. If eviction is necessary, do so after the chunk has been flushed.
     *
     * Traverse from LRU tail toward head.
     *
     * Important: do not mutate the DLL topology while iterating unless we
     * have already captured the next pointer. Otherwise, relinking while
     * iterating can create cycles and trigger the "infinite loop closing
     * library" guard.
     */
    chk = dset_hdr->lru_tail_ptr;

    /* Traverse from LRU tail toward the head
     *
     * Semantics:
     * - evict == false: flush only (do not reorder, do not unlink/free)
     * - evict == true: flush dirty chunks, then evict all chunks
     */

    while (chk) {
        H5SC_chunk_t *prev = chk->prev_ptr;

#if H5SC_DO_SANITY_CHECKS
        assert(chk->magic == H5SC_CHUNK_MAGIC);
#endif

        /* Enforce shell-chunk invariants (H5S_ALL-created, fill-only shells). */
        if (!chk->chunk_obj && !H5_addr_defined(chk->disk_addr)) {
            assert(!chk->dirty_flag);
            assert(chk->cached_chunk_size == 0);

            if (evict_after_flush) {
                if (H5SC__evict_one_chunk(cache, dset, dset_hdr, chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed to evict shell chunk (SCC)");
            }
            chk = prev;
            continue;
        }

        /* Flush if dirty */
        if (chk->dirty_flag) {
            if (H5SC__flush_one_chunk(cache, dset, dset_hdr, chk) < 0) {
                chk->last_op = H5SC_TAG_FLUSH;
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTFLUSH, FAIL, "failed to flush dirty chunk (SCC)");
            }
        }

        chk->last_op = H5SC_TAG_FLUSH;

        /* Evict after flushing (or immediately if already clean) */
        if (evict_after_flush) {
            if (chk->dirty_flag)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "chunk remained dirty after dataset flush");

            if (H5SC__evict_one_chunk(cache, dset, dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed to evict chunk during flush (SCC)");
        }
        chk = prev;
    }

    H5SC__stats_record_dset_flush(cache);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_flush_dset() */

/*-------------------------------------------------------------------------
 * Function: H5SC__account_dset_size_change
 *
 * Purpose:
 *   Function that applies a dataset-local size change to the global SCC
 *     quiescent-size accounting, but only when the dataset is currently
 *     linked into the global dataset LRU.
 *
 *   If the dataset is tracked on the global dataset LRU, the function
 *     adjusts SCC_quiescent_size by the difference between old_size and
 *     new_size. Size increases grow the global quiescent total, while size
 *     decreases reduce it. In sanity-check builds, the function guards
 *     against underflow of the global quiescent-size counter.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state whose global quiescent-size accounting
 *     is to be updated.
 *
 *   H5SC_dset_header_t *dset_hdr:
 *     Pointer to the dataset header whose size change is being reflected in
 *     global cache accounting.
 *
 *   size_t old_size:
 *     Previous accounted size of the dataset.
 *
 *   size_t new_size:
 *     New accounted size of the dataset.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__account_dset_size_change(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, size_t old_size, size_t new_size)
{
    size_t delta     = 0;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset_hdr);

    /* Only tracked datasets contribute to global quiescent-size accounting. */
    if (dset_hdr->next_dset_ptr || dset_hdr->prev_dset_ptr || cache->dset_lru_head_ptr == dset_hdr) {
        if (new_size >= old_size)
            cache->SCC_quiescent_size += (new_size - old_size);
        else {
            delta = old_size - new_size;
#if H5SC_DO_SANITY_CHECKS
            if (delta > cache->SCC_quiescent_size)
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "global quiescent size underflow");
#endif
            cache->SCC_quiescent_size -= delta;
        }
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__account_dset_size_change() */

/*-------------------------------------------------------------------------
 * Function: H5SC__account_chunk_link_change
 *
 * Purpose:
 *   Function that applies the effect of a chunk-linkage change on dataset
 *     and global SCC size accounting. After the caller has modified the
 *     dataset's chunk tracking state (for example, by inserting, removing,
 *     or resizing a chunk), this function compares the dataset's previous
 *     accounted size against its current accounted size and propagates that
 *     delta to global quiescent-size accounting.
 *
 *   The function assumes that dset_hdr->curr_dset_size has already been
 *     updated by the caller to reflect the post-change dataset size. It
 *     then delegates the global accounting update to
 *     H5SC__account_dset_size_change().
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state whose global quiescent-size
 *     accounting may need to be updated.
 *
 *   H5SC_dset_header_t *dset_hdr:
 *     Pointer to the dataset header whose current accounted size reflects
 *     the post-change chunk state.
 *
 *   size_t old_dset_size:
 *     The dataset's accounted size before the caller modified the chunk
 *     linkage or chunk size state.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__account_chunk_link_change(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, size_t old_dset_size)
{
    size_t new_dset_size;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset_hdr);

    /* The caller is expected to have already updated curr_dset_size. */
    new_dset_size = dset_hdr->curr_dset_size;

    if (H5SC__account_dset_size_change(cache, dset_hdr, old_dset_size, new_dset_size) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "failed to account dataset size change");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__account_chunk_link_change() */

/*-------------------------------------------------------------------------
 * Function: H5SC__update_resident_estimate_history
 *
 * Purpose:
 *   Incorporate a successfully observed final resident allocation into the
 *   dataset-local adaptive estimate.
 *
 *   The function maintains an integer exponential moving average with
 *   alpha = 1/8 and a decaying high-water observation. Growth updates the
 *   high-water value immediately; downward decay is considered once every
 *   32 observations. The sample counter saturates at UINT64_MAX.
 *
 *   ACTUAL must be the complete layout-reported resident allocation after
 *   successful materialization or mutation, not an encoded on-disk size.
 *-------------------------------------------------------------------------
 */
static void
H5SC__update_resident_estimate_history(H5SC_dset_header_t *dset_hdr, size_t actual)
{
    size_t ema;
    size_t delta;

    assert(dset_hdr);

    if (dset_hdr->resident_estimate_samples == 0) {
        dset_hdr->resident_estimate_ema     = actual;
        dset_hdr->resident_estimate_high    = actual;
        dset_hdr->resident_estimate_samples = 1;
        return;
    }

    /*
     * Integer EMA with alpha = 1/8. Perform the subtraction form carefully
     * to avoid unsigned underflow.
     */
    ema = dset_hdr->resident_estimate_ema;

    if (actual >= ema) {
        delta = actual - ema;
        ema += (delta + 7) / 8;
    }
    else {
        delta = ema - actual;
        ema -= (delta + 7) / 8;
    }

    dset_hdr->resident_estimate_ema = ema;

    /*
     * Retain a decaying high-water estimate. Update immediately on growth;
     * decay only once every 32 observations.
     */
    if (actual >= dset_hdr->resident_estimate_high)
        dset_hdr->resident_estimate_high = actual;
    else if ((dset_hdr->resident_estimate_samples % 32) == 0) {
        size_t high  = dset_hdr->resident_estimate_high;
        size_t decay = high / 32;

        if (decay == 0 && high > actual)
            decay = 1;

        high -= MIN(decay, high - actual);
        dset_hdr->resident_estimate_high = MAX(high, actual);
    }

    if (dset_hdr->resident_estimate_samples < UINT64_MAX)
        dset_hdr->resident_estimate_samples++;
} /* end H5SC__update_resident_estimate_history() */

/*-------------------------------------------------------------------------
 * Function: H5SC__chunk_reclaim_contribution
 *
 * Purpose:
 *   Compute CHUNK's current contribution to the dataset-local and cache-wide
 *   active-pressure reclaimability counters.
 *
 *   Only linked, unpinned chunks with a nonzero cached resident size
 *   contribute. Clean and dirty resident bytes are returned separately.
 *   Dataset-local min_dset_size is intentionally not applied because stored
 *   reclaimability counters represent active-pressure policy.
 *-------------------------------------------------------------------------
 */
static inline H5SC_reclaim_contrib_t
H5SC__chunk_reclaim_contribution(const H5SC_dset_header_t *dset_hdr, const H5SC_chunk_t *chunk)
{
    H5SC_reclaim_contrib_t contribution = {0, 0};
    bool                   linked;

    assert(dset_hdr);
    assert(chunk);

    linked = chunk->next_ptr || chunk->prev_ptr || dset_hdr->lru_head_ptr == chunk;

    if (!linked)
        return contribution;

    if (chunk->chunk_counter > 0)
        return contribution;

    if (chunk->cached_chunk_size == 0)
        return contribution;

    if (chunk->dirty_flag)
        contribution.dirty_bytes = chunk->cached_chunk_size;
    else
        contribution.clean_bytes = chunk->cached_chunk_size;

    return contribution;
} /* end H5SC__chunk_reclaim_contribution() */

/*-------------------------------------------------------------------------
 * Function: H5SC__apply_reclaim_transition
 *
 * Purpose:
 *   Apply the difference between a chunk's BEFORE and AFTER reclaimability
 *   contributions to both dataset-local and cache-wide stored counters.
 *
 *   Callers must capture BEFORE prior to mutating linkage, pin, dirty, or
 *   resident-size state and capture AFTER after completing that mutation.
 *   In sanity-check builds, removal of an unaccounted contribution fails
 *   before counters are modified.
 *
 * Return:
 *   SUCCEED when both counter levels are updated;
 *   FAIL when sanity checking detects inconsistent prior accounting.
 *-------------------------------------------------------------------------
 */
static inline herr_t
H5SC__apply_reclaim_transition(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_reclaim_contrib_t before,
                               H5SC_reclaim_contrib_t after)
{
    assert(cache);
    assert(dset_hdr);

#if H5SC_DO_SANITY_CHECKS
    if (before.clean_bytes > dset_hdr->reclaimable_clean_bytes ||
        before.clean_bytes > cache->reclaimable_clean_bytes)
        return FAIL;

    if (before.dirty_bytes > dset_hdr->reclaimable_dirty_bytes ||
        before.dirty_bytes > cache->reclaimable_dirty_bytes)
        return FAIL;
#endif

    dset_hdr->reclaimable_clean_bytes -= before.clean_bytes;
    dset_hdr->reclaimable_dirty_bytes -= before.dirty_bytes;

    cache->reclaimable_clean_bytes -= before.clean_bytes;
    cache->reclaimable_dirty_bytes -= before.dirty_bytes;

    H5SC__saturating_add_size(&dset_hdr->reclaimable_clean_bytes, after.clean_bytes);
    H5SC__saturating_add_size(&dset_hdr->reclaimable_dirty_bytes, after.dirty_bytes);

    H5SC__saturating_add_size(&cache->reclaimable_clean_bytes, after.clean_bytes);
    H5SC__saturating_add_size(&cache->reclaimable_dirty_bytes, after.dirty_bytes);

    return SUCCEED;
} /* end H5SC__apply_reclaim_transition() */

/*-------------------------------------------------------------------------
 * Function: H5SC__chunk_set_dirty
 *
 * Purpose:
 *   Change a resident chunk's dirty state while maintaining dataset-local
 *   and cache-wide active-pressure reclaimability counters.
 *
 *   Pinned chunks make no current reclaimability contribution, but their
 *   dirty state is still changed so that the correct contribution is added
 *   when the final request pin is released.
 *-------------------------------------------------------------------------
 */
static inline herr_t
H5SC__chunk_set_dirty(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk, bool dirty)
{
    H5SC_reclaim_contrib_t before;
    H5SC_reclaim_contrib_t after;
    bool                   old_dirty;

    assert(cache);
    assert(dset_hdr);
    assert(chunk);

    if (chunk->dirty_flag == dirty)
        return SUCCEED;

    before    = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);
    old_dirty = chunk->dirty_flag;

    chunk->dirty_flag = dirty;

    after = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    if (H5SC__apply_reclaim_transition(cache, dset_hdr, before, after) < 0) {
        chunk->dirty_flag = old_dirty;
        return FAIL;
    }

    return SUCCEED;
} /* end H5SC__chunk_set_dirty() */

/*-------------------------------------------------------------------------
 * Function: H5SC__candidate_is_reclaimable
 *
 * Purpose:
 *   Determine whether CHUNK can contribute resident bytes to a reclamation
 *   operation under POLICY.
 *
 *   Both policies reject pinned chunks, zero-byte shells, and chunks whose
 *   recorded size exceeds their dataset's total resident accounting.
 *
 *   H5SC_RECLAIM_ACTIVE treats SCC_active_limit as a hard operational
 *   ceiling and therefore permits reclaiming a chunk even when doing so
 *   reduces the dataset below min_dset_size.
 *
 *   H5SC_RECLAIM_QUIESCENT treats min_dset_size as a steady-state retention
 *   target and rejects a chunk when reclaiming it would reduce the dataset
 *   below that target.
 *
 * Return:
 *   true if the chunk is eligible under POLICY; otherwise false.
 *-------------------------------------------------------------------------
 */
static bool
H5SC__candidate_is_reclaimable(const H5SC_dset_header_t *dset_hdr, const H5SC_chunk_t *chunk,
                               H5SC_reclaim_policy_t policy)
{
    size_t chunk_size;

    assert(dset_hdr);
    assert(chunk);
    assert(policy == H5SC_RECLAIM_ACTIVE || policy == H5SC_RECLAIM_QUIESCENT);

    if (chunk->chunk_counter > 0)
        return false;

    chunk_size = chunk->cached_chunk_size;

    /* Removing a shell does not satisfy a byte-reclamation request. */
    if (chunk_size == 0)
        return false;

    if (chunk_size > dset_hdr->curr_dset_size)
        return false;

    if (policy == H5SC_RECLAIM_ACTIVE)
        return true;

    if (dset_hdr->curr_dset_size <= dset_hdr->min_dset_size)
        return false;

    return dset_hdr->curr_dset_size - chunk_size >= dset_hdr->min_dset_size;
} /* end H5SC__candidate_is_reclaimable() */

/*-------------------------------------------------------------------------
 * Function: H5SC__select_clean_candidate
 *
 * Purpose:
 *   Select the tailmost clean chunk from DSET_HDR that is reclaimable under
 *   POLICY. Active-pressure selection may reduce the dataset below
 *   min_dset_size; quiescent selection preserves that retention target.
 *
 * Return:
 *   true and sets *CANDIDATE when an eligible chunk is found;
 *   false and sets *CANDIDATE to NULL otherwise.
 *-------------------------------------------------------------------------
 */
static bool
H5SC__select_clean_candidate(H5SC_dset_header_t *dset_hdr, H5SC_reclaim_policy_t policy,
                             H5SC_chunk_t **candidate)
{
    assert(dset_hdr);
    assert(candidate);
    assert(policy == H5SC_RECLAIM_ACTIVE || policy == H5SC_RECLAIM_QUIESCENT);

    *candidate = NULL;

    if (!dset_hdr->dset)
        return false;

    for (H5SC_chunk_t *chunk = dset_hdr->lru_tail_ptr; chunk; chunk = chunk->prev_ptr) {
        if (chunk->dirty_flag)
            continue;

        if (!H5SC__candidate_is_reclaimable(dset_hdr, chunk, policy))
            continue;

        *candidate = chunk;
        return true;
    }

    return false;
}

/*-------------------------------------------------------------------------
 * Function: H5SC__select_dirty_candidate
 *
 * Purpose:
 *   Select the tailmost dirty chunk from DSET_HDR that may be reclaimed by
 *   flush-and-evict under POLICY. Active-pressure selection may reduce the
 *   dataset below min_dset_size; quiescent selection preserves that target.
 *
 * Return:
 *   true and sets *CANDIDATE when an eligible chunk is found;
 *   false and sets *CANDIDATE to NULL otherwise.
 *-------------------------------------------------------------------------
 */
static bool
H5SC__select_dirty_candidate(H5SC_dset_header_t *dset_hdr, H5SC_reclaim_policy_t policy,
                             H5SC_chunk_t **candidate)
{
    assert(dset_hdr);
    assert(candidate);
    assert(policy == H5SC_RECLAIM_ACTIVE || policy == H5SC_RECLAIM_QUIESCENT);

    *candidate = NULL;

    if (!dset_hdr->dset)
        return false;

    for (H5SC_chunk_t *chunk = dset_hdr->lru_tail_ptr; chunk; chunk = chunk->prev_ptr) {
        if (!chunk->dirty_flag)
            continue;

        if (!H5SC__candidate_is_reclaimable(dset_hdr, chunk, policy))
            continue;

        *candidate = chunk;
        return true;
    }

    return false;
}

#if H5SC_DO_SANITY_CHECKS
/*-------------------------------------------------------------------------
 * Function: H5SC__calc_reclaimable_bytes
 *
 * Purpose:
 *   Independently recompute resident bytes reclaimable under POLICY by
 *   scanning dataset and chunk LRUs.
 *
 *   This function is a sanity-check oracle only. Production admission and
 *   active-pressure feasibility decisions use the stored cache-wide clean
 *   and dirty reclaimability counters.
 *
 *   H5SC_RECLAIM_ACTIVE ignores min_dset_size and counts all unpinned,
 *   nonzero resident chunks. H5SC_RECLAIM_QUIESCENT simulates clean-first,
 *   dirty-second reclamation while preserving min_dset_size.
 *
 * Return:
 *   Recomputed reclaimable resident bytes, saturating at SIZE_MAX.
 *-------------------------------------------------------------------------
 */
static size_t
H5SC__calc_reclaimable_bytes(const H5SC_t *cache, H5SC_reclaim_policy_t policy)
{
    size_t reclaimable = 0;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(policy == H5SC_RECLAIM_ACTIVE || policy == H5SC_RECLAIM_QUIESCENT);

    for (const H5SC_dset_header_t *dset_hdr = cache->dset_lru_tail_ptr; dset_hdr;
         dset_hdr                           = dset_hdr->prev_dset_ptr) {
        size_t remaining = dset_hdr->curr_dset_size;

        /*
         * Active-pressure reclamation ignores min_dset_size and counts every
         * eligible clean or dirty resident chunk.
         */
        if (policy == H5SC_RECLAIM_ACTIVE) {
            for (const H5SC_chunk_t *chunk = dset_hdr->lru_tail_ptr; chunk; chunk = chunk->prev_ptr) {
                size_t chunk_size = chunk->cached_chunk_size;

                if (chunk->chunk_counter > 0)
                    continue;

                if (chunk_size == 0)
                    continue;

                if (chunk_size > dset_hdr->curr_dset_size)
                    continue;

                H5SC__saturating_add_size(&reclaimable, chunk_size);
            }

            continue;
        }

        /*
         * Quiescent reclamation preserves the dataset-local retention target.
         */
        if (remaining <= dset_hdr->min_dset_size)
            continue;

        /* Simulate clean eviction first. */
        for (const H5SC_chunk_t *chunk = dset_hdr->lru_tail_ptr; chunk; chunk = chunk->prev_ptr) {
            size_t chunk_size = chunk->cached_chunk_size;

            if (chunk->chunk_counter > 0 || chunk->dirty_flag || chunk_size == 0)
                continue;

            if (chunk_size > remaining)
                continue;

            if (remaining - chunk_size < dset_hdr->min_dset_size)
                continue;

            remaining -= chunk_size;

            H5SC__saturating_add_size(&reclaimable, chunk_size);
        }

        /* Then simulate dirty flush-and-evict. */
        for (const H5SC_chunk_t *chunk = dset_hdr->lru_tail_ptr; chunk; chunk = chunk->prev_ptr) {
            size_t chunk_size = chunk->cached_chunk_size;

            if (chunk->chunk_counter > 0 || !chunk->dirty_flag || chunk_size == 0)
                continue;

            if (chunk_size > remaining)
                continue;

            if (remaining - chunk_size < dset_hdr->min_dset_size)
                continue;

            remaining -= chunk_size;

            H5SC__saturating_add_size(&reclaimable, chunk_size);
        }
    }

    return reclaimable;
} /* end H5SC__calc_reclaimable_bytes() */

/*-------------------------------------------------------------------------
 * Function: H5SC__verify_cached_reclaimability
 *
 * Purpose:
 *   Independently recompute dataset-local and cache-wide active-pressure
 *   reclaimability and compare those values with the stored counters.
 *
 *   This function is a sanity-check oracle only. Production admission uses
 *   the stored counters and does not perform this full-cache scan.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__verify_cached_reclaimability(const H5SC_t *cache)
{
    size_t cached_total;
    size_t recomputed_total;
    size_t cache_clean = 0;
    size_t cache_dirty = 0;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    for (const H5SC_dset_header_t *dset_hdr = cache->dset_lru_head_ptr; dset_hdr;
         dset_hdr                           = dset_hdr->next_dset_ptr) {
        size_t dset_clean = 0;
        size_t dset_dirty = 0;

        for (const H5SC_chunk_t *chunk = dset_hdr->lru_head_ptr; chunk; chunk = chunk->next_ptr) {
            if (chunk->chunk_counter > 0 || chunk->cached_chunk_size == 0)
                continue;

            if (chunk->dirty_flag)
                H5SC__saturating_add_size(&dset_dirty, chunk->cached_chunk_size);
            else
                H5SC__saturating_add_size(&dset_clean, chunk->cached_chunk_size);
        }

        if (dset_clean != dset_hdr->reclaimable_clean_bytes)
            return FAIL;

        if (dset_dirty != dset_hdr->reclaimable_dirty_bytes)
            return FAIL;

        H5SC__saturating_add_size(&cache_clean, dset_clean);
        H5SC__saturating_add_size(&cache_dirty, dset_dirty);
    }

    if (cache_clean != cache->reclaimable_clean_bytes || cache_dirty != cache->reclaimable_dirty_bytes)
        return FAIL;

    cached_total     = H5SC__cached_active_reclaimable(cache);
    recomputed_total = H5SC__calc_reclaimable_bytes(cache, H5SC_RECLAIM_ACTIVE);

    if (cached_total != recomputed_total)
        return FAIL;

    return SUCCEED;
} /* end H5SC__verify_cached_reclaimability() */

/*-------------------------------------------------------------------------
 * Function: H5SC__verify_clean_candidate
 *
 * Purpose:
 *   Verify that CANDIDATE is the clean chunk that would be selected from
 *   DSET_HDR under the same reclamation POLICY used by the caller.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__verify_clean_candidate(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *candidate,
                             H5SC_reclaim_policy_t policy)
{
    H5SC_chunk_t *expected = NULL;

    assert(dset_hdr);
    assert(candidate);

    if (!H5SC__select_clean_candidate(dset_hdr, policy, &expected))
        return FAIL;

    return expected == candidate ? SUCCEED : FAIL;
}

/*-------------------------------------------------------------------------
 * Function: H5SC__verify_dirty_candidate
 *
 * Purpose:
 *   Verify that CANDIDATE is the dirty chunk that would be selected from
 *   DSET_HDR under the same reclamation POLICY used by the caller.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__verify_dirty_candidate(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *candidate,
                             H5SC_reclaim_policy_t policy)
{
    H5SC_chunk_t *expected = NULL;

    assert(dset_hdr);
    assert(candidate);

    if (!H5SC__select_dirty_candidate(dset_hdr, policy, &expected))
        return FAIL;

    return expected == candidate ? SUCCEED : FAIL;
}
#endif

/*-------------------------------------------------------------------------
 * Function: H5SC__trim_to_quiescent_limit
 *
 * Purpose:
 *   Reduce SCC_quiescent_size until it is no greater than
 *   SCC_quiescent_limit.
 *
 *   This is a steady-state reclamation operation. Candidate selection uses
 *   H5SC_RECLAIM_QUIESCENT and therefore preserves each dataset's
 *   min_dset_size retention target. Clean chunks are evicted before dirty
 *   chunks; a dirty candidate is flushed before eviction.
 *
 *   The function does not perform a preliminary full-cache reclaimability
 *   scan. It traverses quiescent-policy candidates directly and returns FAIL
 *   if the cache remains above the limit after all eligible candidates have
 *   been exhausted.
 *
 *   Because quiescent candidate selection preserves min_dset_size, the cache
 *   may remain above its quiescent limit when aggregate dataset retention
 *   targets exceed that limit.
 *
 * Return:
 *   SUCCEED when the cache reaches the quiescent limit;
 *   FAIL when the target cannot be reached or flush/eviction fails.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__trim_to_quiescent_limit(H5SC_t *cache)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    if (cache->SCC_quiescent_size <= cache->SCC_quiescent_limit)
        HGOTO_DONE(SUCCEED);

    for (H5SC_dset_header_t *hdr                                            = cache->dset_lru_tail_ptr;
         hdr && cache->SCC_quiescent_size > cache->SCC_quiescent_limit; hdr = hdr->prev_dset_ptr) {
        H5SC_chunk_t *candidate = NULL;

        if (!hdr->dset)
            continue;

        while (cache->SCC_quiescent_size > cache->SCC_quiescent_limit &&
               H5SC__select_clean_candidate(hdr, H5SC_RECLAIM_QUIESCENT, &candidate)) {
            candidate->last_op = H5SC_TAG_EVICT;

            if (H5SC__evict_one_chunk(cache, hdr->dset, hdr, candidate) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed clean SCC quiescent eviction");
        }

        while (cache->SCC_quiescent_size > cache->SCC_quiescent_limit &&
               H5SC__select_dirty_candidate(hdr, H5SC_RECLAIM_QUIESCENT, &candidate)) {
            candidate->last_op = H5SC_TAG_FLUSH_DIRTY;

            if (H5SC__flush_one_chunk(cache, hdr->dset, hdr, candidate) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTFLUSH, FAIL, "failed dirty SCC quiescent flush");

            candidate->last_op = H5SC_TAG_EVICT;

            if (H5SC__evict_one_chunk(cache, hdr->dset, hdr, candidate) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed dirty SCC quiescent eviction");
        }
    }

    if (cache->SCC_quiescent_size > cache->SCC_quiescent_limit)
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "SCC remains above quiescent limit");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__trim_to_quiescent_limit() */

/*-------------------------------------------------------------------------
 * Function: H5SC__active_reclaim_required
 *
 * Purpose:
 *   Determine whether active-pressure reclamation should continue while
 *   preparing an SCC admission.
 *
 *   For an ordinary admission, reclamation remains necessary while admitting
 *   BYTES_NEEDED additional resident bytes would cause the cache's current
 *   resident size to exceed SCC_active_limit.
 *
 *   For an oversized single-chunk admission, the normal active limit is
 *   temporarily bypassed because the indivisible chunk cannot fit within the
 *   effective budget. In this mode, reclamation continues until no clean or
 *   dirty resident bytes remain reclaimable under H5SC_RECLAIM_ACTIVE.
 *
 *   This helper does not mutate cache state and does not authorize an
 *   oversized admission. The caller is responsible for establishing that the
 *   current operation contains exactly one indivisible chunk and for restoring
 *   ordinary active-limit enforcement after that chunk is unpinned.
 *
 * Return:
 *   true if active-pressure reclamation should continue;
 *   false otherwise.
 *-------------------------------------------------------------------------
 */
static inline bool
H5SC__active_reclaim_required(const H5SC_t *cache, size_t bytes_needed, bool oversized)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    if (oversized)
        return H5SC__cached_active_reclaimable(cache) > 0;

    return H5SC__space_exceeded(cache->SCC_quiescent_size, bytes_needed, cache->SCC_active_limit);
} /* end H5SC__active_reclaim_required() */

/*-------------------------------------------------------------------------
 * Function: H5SC__ensure_space
 *
 * Purpose:
 *   Ensure that BYTES_NEEDED additional resident bytes can be admitted under
 *   the active SCC policy.
 *
 *   During ordinary admission, SCC_active_limit remains a hard operational
 *   ceiling. Candidate selection uses H5SC_RECLAIM_ACTIVE and may therefore
 *   reduce a dataset below min_dset_size. The latter is a quiescent-retention
 *   preference rather than an active-pressure reservation.
 *
 *   For an ordinary admission, the function first determines whether enough
 *   active-policy bytes are reclaimable to satisfy the request. It then walks
 *   datasets from the global LRU tail, evicting eligible clean chunks before
 *   flushing and evicting eligible dirty chunks.
 *
 *   When ALLOW_OVERSIZED_SINGLE_CHUNK is true, the caller has established
 *   that the current batch contains exactly one indivisible chunk. If that
 *   chunk's incremental resident-size estimate exceeds the current effective
 *   budget, the function enables the oversized exception. In that mode, all
 *   currently reclaimable active-policy bytes are reclaimed before the chunk
 *   is admitted, even though the resulting temporary working set may exceed
 *   SCC_active_limit.
 *
 *   The exception does not modify SCC_active_limit. The caller must process
 *   only the single oversized chunk and must restore ordinary active-limit
 *   enforcement immediately after that chunk is unpinned.
 *
 * Return:
 *   SUCCEED when the requested bytes may be admitted;
 *   FAIL when ordinary admission is infeasible, active-pressure reclamation
 *   fails, or an oversized request cannot be prepared safely.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__ensure_space(H5SC_t *cache, size_t bytes_needed, bool allow_oversized_single_chunk)
{
    bool   oversized = false;
    size_t reclaimable;
    size_t required  = 0;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    /*
     * The caller authorizes the exception, but it is needed only when this
     * admission cannot fit within the current effective budget.
     */
    oversized = allow_oversized_single_chunk && bytes_needed > H5SC__effective_budget(cache);

    /*
     * A request larger than the configured active limit is invalid unless it
     * is the explicitly authorized one-chunk oversized admission.
     */
    if (bytes_needed > cache->SCC_active_limit && !oversized)
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "single SCC request exceeds active limit");

    /*
     * Ordinary admission can return immediately when the requested
     * incremental bytes fit without reclamation.
     *
     * Oversized admission deliberately does not return here. It must reclaim
     * every currently reclaimable active-policy byte before dispatch.
     */
    if (!oversized && !H5SC__space_exceeded(cache->SCC_quiescent_size, bytes_needed, cache->SCC_active_limit))
        HGOTO_DONE(SUCCEED);

    reclaimable = H5SC__cached_active_reclaimable(cache);

    if (!oversized) {
        /*
         * At this point:
         *
         *     SCC_quiescent_size + bytes_needed > SCC_active_limit
         *
         * and bytes_needed <= SCC_active_limit, so the subtraction below is
         * well-defined.
         */
        required = cache->SCC_quiescent_size - (cache->SCC_active_limit - bytes_needed);

        /*
         * Reject an infeasible ordinary admission before mutating cache
         * topology or flushing dirty chunks.
         */
        if (reclaimable < required)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "insufficient reclaimable SCC space");
    }

    /*
     * Drain one dataset at a time, beginning with the global LRU tail.
     * Preserve global dataset-LRU topology throughout.
     *
     * Ordinary mode stops after sufficient space has been recovered.
     * Oversized mode continues until no active-policy bytes remain
     * reclaimable.
     */
    for (H5SC_dset_header_t *hdr                                                   = cache->dset_lru_tail_ptr;
         hdr && H5SC__active_reclaim_required(cache, bytes_needed, oversized); hdr = hdr->prev_dset_ptr) {
        H5SC_chunk_t *candidate = NULL;

        if (!hdr->dset)
            continue;

        /*
         * Phase 1: evict eligible clean candidates from this dataset.
         */
        while (H5SC__active_reclaim_required(cache, bytes_needed, oversized) &&
               H5SC__select_clean_candidate(hdr, H5SC_RECLAIM_ACTIVE, &candidate)) {
#if H5SC_DO_SANITY_CHECKS
            if (H5SC__verify_clean_candidate(hdr, candidate, H5SC_RECLAIM_ACTIVE) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_NOT_TAILMOST, FAIL, "incorrect clean eviction candidate");
#endif

            candidate->last_op = H5SC_TAG_EVICT;

            if (H5SC__evict_one_chunk(cache, hdr->dset, hdr, candidate) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed to evict clean SCC chunk");
        }

        /*
         * Phase 2: flush and evict eligible dirty candidates from this
         * dataset before advancing to the next dataset.
         */
        while (H5SC__active_reclaim_required(cache, bytes_needed, oversized) &&
               H5SC__select_dirty_candidate(hdr, H5SC_RECLAIM_ACTIVE, &candidate)) {
#if H5SC_DO_SANITY_CHECKS
            if (H5SC__verify_dirty_candidate(hdr, candidate, H5SC_RECLAIM_ACTIVE) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_NOT_TAILMOST, FAIL, "incorrect dirty eviction candidate");
#endif

            candidate->last_op = H5SC_TAG_FLUSH_DIRTY;

            if (H5SC__flush_one_chunk(cache, hdr->dset, hdr, candidate) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTFLUSH, FAIL, "failed to flush dirty SCC chunk");

            candidate->last_op = H5SC_TAG_EVICT;

            if (H5SC__evict_one_chunk(cache, hdr->dset, hdr, candidate) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed to evict flushed SCC chunk");
        }
    }

    if (oversized) {
        /*
         * Cached reclaimability must agree with candidate traversal. Any
         * remaining contribution means the SCC failed to prepare the minimum
         * resident working set for the oversized chunk.
         */
        if (H5SC__cached_active_reclaimable(cache) != 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                        "unable to reclaim available SCC space before oversized admission");
    }
    else if (H5SC__space_exceeded(cache->SCC_quiescent_size, bytes_needed, cache->SCC_active_limit)) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "SCC reclamation did not produce sufficient space");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__ensure_space() */

/*-------------------------------------------------------------------------
 * Function: H5SC__calc_inc_for_chunk
 *
 * Purpose:
 *   Function that computes the incremental number of bytes required to
 *     materialize or grow an SCC chunk under the current cache accounting
 *     model. The returned value represents the additional resident bytes
 *     needed beyond the chunk's current cached size.
 *
 *   The computation follows:
 *
 *       inc = max(0, desired - chk->cached_chunk_size)
 *
 *   where desired is the target resident size for the operation. If the
 *     caller supplies desired == 0, the function falls back to the
 *     dataset's dense logical chunk size as the cold-dataset fallback
 *     estimate. This fallback is an admission heuristic and is not a formal
 *     upper bound on decoded resident allocation.
 *
 * Inputs:
 *   const H5D_t *dset:
 *     Pointer to the dataset whose chunk layout information is used when a
 *     fallback size estimate is required.
 *
 *   const H5SC_chunk_t *chk:
 *     Pointer to the SCC chunk whose current cached size is used as the
 *     baseline for the incremental-size calculation.
 *
 *   size_t desired:
 *     Target resident size for the operation. If zero, the dataset chunk
 *     size is used as a conservative fallback.
 *
 * Return:
 *   The additional number of bytes required beyond chk->cached_chunk_size
 *   to satisfy the requested target size.
 *-------------------------------------------------------------------------
 */

static size_t
H5SC__calc_inc_for_chunk(const H5D_t *dset, const H5SC_chunk_t *chk, size_t desired)
{
    size_t fallback = 0;

    assert(dset);
    assert(chk);

    if (desired == 0) {
        /* Use dense logical chunk size as the cold-dataset fallback estimate. */
        fallback = (size_t)dset->shared->layout.u.struct_chunk.size;
        desired  = fallback;
    }

    if (desired > chk->cached_chunk_size)
        return (desired - chk->cached_chunk_size);
    else
        return 0;
} /* end H5SC__calc_inc_for_chunk() */

/*-------------------------------------------------------------------------
 * Function: H5SC__invoke_read_dset_batched
 *
 * Purpose:
 *   Function that executes a dataset read through SCC using invoke-layer
 *     batching. The function partitions the current selected-chunk window
 *     into one or more sub-batches sized to fit within the effective cache
 *     budget, performs any necessary pre-read eviction, and invokes
 *     H5SC_read() on each batch in sequence.
 *
 *   The effective budget for each batch is computed from the current SCC
 *     active/quiescent accounting plus the amount of clean cache space that
 *     can be freed immediately. For each provisional batch window, the
 *     function ensures lookup metadata is available, estimates the
 *     incremental resident bytes required for each candidate chunk, greedily
 *     extends the batch while staying within budget, and evicts before
 *     dispatching the read.
 *
 *   The function uses selection-window semantics throughout batching:
 *     sel_chunks remains a stable base pointer, while sel_start and
 *     num_sel_chunks define the active subrange for the current batch.
 *     On exit, the original selection window is restored.
 *
 *   When H5SC_BATCH_INTO_SINGLE_CHUNK is defined, the function instead
 *     processes one selected chunk at a time for debugging. That path is
 *     intended for investigation and validation and should preserve its
 *     existing commented debugging code.
 *
 *   On failure, the function performs conservative pin unwinding for any
 *     selected chunks that were pinned during I/O setup but not yet
 *     unpinned by the read path, ensuring chunk_counter is decremented
 *     at most once for each remaining pinned selection entry.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state used for budgeting, eviction, and
 *     read execution.
 *
 *   H5D_dset_io_info_t *dset_info:
 *     Pointer to the dataset I/O context for the read being processed.
 *
 *   H5SC_dset_header_t *dset_hdr:
 *     Pointer to the SCC dataset header associated with this I/O request.
 *     Its io_info field supplies the selected-chunk window to batch.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__invoke_read_dset_batched(H5SC_t *cache, H5D_dset_io_info_t *dset_info, H5SC_dset_header_t *dset_hdr)
{
    H5SC_io_info_t      *io;
    H5SC_io_sel_chunk_t *saved_sel;
    size_t               saved_start;
    size_t               saved_n;
    const size_t        *saved_order;
    size_t               saved_resident_n;
    size_t               processed = 0;

    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset_info);
    assert(dset_hdr);
    io = dset_hdr->io_info;
    assert(io);

    saved_sel        = io->sel_chunks;
    saved_order      = io->sel_order;
    saved_start      = io->sel_start;
    saved_n          = io->num_sel_chunks;
    saved_resident_n = io->num_resident_sel_chunks;

    if (saved_n == 0)
        HGOTO_DONE(SUCCEED);

#ifdef H5SC_BATCH_INTO_SINGLE_CHUNK
    /* Process each selected chunk individually (debugging mode). */
    size_t i = 0;
    for (i = 0; i < saved_n; i++) {
        io->sel_chunks     = saved_sel;
        io->sel_order      = saved_order ? &saved_order[i] : NULL;
        io->sel_start      = saved_order ? saved_start : saved_start + i;
        io->num_sel_chunks = 1;

#ifdef DO_SANITY_CHECKS
        H5SC_test_read_batch_calls++;
        H5SC_test_read_last_batch_len = io->num_sel_chunks;
        if (io->num_sel_chunks > H5SC_test_read_max_batch_len)
            H5SC_test_read_max_batch_len = io->num_sel_chunks;
#endif

        if (H5SC_read(cache, dset_info) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to process batched SCC read");
        }

        H5SC__stats_record_batch_read(cache);
        H5SC__stats_record_batch_read_len(cache, 1);
    }
    goto done;
#endif

    /*
     * Full batching:
     *  - Build the largest batch that fits within the current budget
     *  - Evict before dispatching the batch
     *  - Repeat until the full selection window is processed
     */

    while (processed < saved_n) {
        size_t budget;
        size_t phase_end;

        if (processed < saved_resident_n)
            phase_end = saved_resident_n;
        else
            phase_end = saved_n;

        budget = H5SC__effective_budget(cache);

        /* Configure io_info to expose this provisional window. */
        io->sel_order      = saved_order ? &saved_order[processed] : NULL;
        io->sel_start      = saved_order ? saved_start : saved_start + processed;
        io->num_sel_chunks = phase_end - processed;

        /* Populate lookup metadata for the provisional window. */
        if (H5SC__lookup_cache_misses(dset_info->dset, io) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTGET, FAIL, "invoke read: miss lookup caching failed");

        /*
         * Access an entry relative to the current provisional batch.
         *
         * sel_chunks/base always points to the beginning of the persistent
         * H5SC_io_sel_chunk_t allocation. saved_start identifies the beginning
         * of the caller's original active selection window, while batch_start
         * identifies the beginning of the current batch relative to that window.
         *
         * Therefore:
         *
         *     absolute index = saved_start + batch_start + j
         *
         * Keep this macro local to this function and undefine it before leaving
         * its lexical use region so that a different SEL() indexing convention
         * cannot accidentally be reused elsewhere in this translation unit.
         */

        size_t batch_len       = 0;
        size_t batch_need      = 0;
        bool   oversized_batch = false;

        /* Greedily extend batch while staying within budget */
        for (size_t j = 0; j < io->num_sel_chunks; j++) {
            H5SC_io_sel_chunk_t *sel     = H5SC__io_sel_at(io, j);
            H5SC_chunk_t        *chk     = sel->cached_chunk;
            size_t               desired = 0;
            size_t               inc     = 0;

            assert(chk);

            /* Estimate additional resident bytes needed for this candidate chunk. */
            if (!chk->chunk_obj) {

                H5SC_size_est_source_t source;

                if (H5SC__estimate_resident_size(dset_hdr, dset_info->dset, sel, false, 0, &source,
                                                 &desired) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANTGETSIZE, FAIL,
                                "invoke read: unable to estimate resident chunk size");

                sel->estimate_resident_size = desired;
                sel->estimate_source        = source;
                sel->estimate_pending       = true;

                /*
                 * IMPORTANT: keep budget semantics identical to the core SCC code.
                 * This handles placeholder/shell accounting correctly.
                 */
                inc = H5SC__calc_inc_for_chunk(dset_info->dset, chk, desired);
            }

            if (inc > budget) {
                /*
                 * Dispatch any ordinary batch already under construction. This
                 * candidate will become the first candidate in the next iteration and
                 * will then be admitted as an oversized one-chunk batch.
                 */
                if (batch_len > 0)
                    break;

                /*
                 * The selection cannot be divided below this one chunk. Admit it as a
                 * one-entry oversized batch after reclaiming everything currently
                 * reclaimable under the active policy.
                 */
                batch_need      = inc;
                batch_len       = 1;
                oversized_batch = true;
                break;
            }

            if (batch_need > budget || inc > budget - batch_need) {
                break;
            }

            batch_need += inc;
            batch_len++;
        }

        assert(batch_len > 0);

        /* Evict before dispatching the batch. batch_need is incremental only. */
        if (batch_need > 0) {
            if (H5SC__ensure_space(cache, batch_need, oversized_batch) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                            "invoke read: unable to prepare SCC space for batch");
        }

        if (oversized_batch)
            H5SC__report_oversized_admission(cache, batch_need);

        /* Finalize the active window for this batch. */
        io->sel_chunks     = saved_sel;
        io->sel_order      = saved_order ? &saved_order[processed] : NULL;
        io->sel_start      = saved_order ? saved_start : saved_start + processed;
        io->num_sel_chunks = batch_len;

        if (H5SC_read(cache, dset_info) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "invoke read: H5SC_read failed");

        if (oversized_batch) {
            /*
             * The oversized chunk is now unpinned. Return to the configured active
             * limit before constructing another batch.
             */
            if (H5SC__ensure_space(cache, 0, false) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                            "invoke read: unable to restore configured active limit "
                            "after oversized chunk");
        }

        H5SC__stats_record_batch_read(cache);
        H5SC__stats_record_batch_read_len(cache, batch_len);

        processed += batch_len;
    }

done:
    /*
     * Restore the complete request window before inspecting or unwinding
     * selection-owned pins. This makes H5SC__io_sel_at() resolve indices
     * against the original selection and ordering arrays rather than the
     * final active batch.
     */
    if (dset_hdr && dset_hdr->io_info) {
        dset_hdr->io_info->sel_chunks              = saved_sel;
        dset_hdr->io_info->sel_order               = saved_order;
        dset_hdr->io_info->sel_start               = saved_start;
        dset_hdr->io_info->num_sel_chunks          = saved_n;
        dset_hdr->io_info->num_resident_sel_chunks = saved_resident_n;

        /*
         * A successful batched read must have released every selection-owned
         * pin. Report an ownership leak instead of silently correcting it.
         */
        if (ret_value >= 0) {
            for (size_t j = 0; j < saved_n; j++) {
                H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(dset_hdr->io_info, j);

                if (sel->pin_held) {
                    HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "batched read completed with an owned pin");
                    break;
                }
            }
        }

        /*
         * Unwind pins only after the operation has failed. This includes a
         * successful processing path converted to failure by the ownership
         * check above.
         */
        if (ret_value < 0) {
            for (size_t j = 0; j < saved_n; j++) {
                H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(dset_hdr->io_info, j);
                H5SC_chunk_t        *chk = sel->cached_chunk;

                if (sel->pin_held) {
                    if (!chk || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                        HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                    "batched read cleanup encountered an invalid owned pin");
                    else {
                        sel->pin_held = false;
                        chk->last_op  = H5SC_TAG_UNPIN_READ_DONE_ERROR;
                    }
                }
            }
        }
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__invoke_read_dset_batched() */

/*-------------------------------------------------------------------------
 * Function: H5SC__invoke_write_dset_batched
 *
 * Purpose:
 *   Function that executes a dataset write through SCC using invoke-layer
 *     batching. The function partitions the current selected-chunk window
 *     into one or more sub-batches sized to fit within the effective cache
 *     budget, performs any necessary pre-write eviction, and invokes
 *     H5SC_write() on each batch in sequence.
 *
 *   The effective budget for each batch is computed from the current SCC
 *     active/quiescent accounting plus the amount of clean cache space that
 *     can be freed immediately. For each provisional batch window, the
 *     function ensures lookup metadata is available, estimates the
 *     incremental resident bytes required for each candidate chunk, greedily
 *     extends the batch while staying within budget, and evicts before
 *     dispatching the write.
 *
 *   The function uses selection-window semantics throughout batching:
 *     sel_chunks remains a stable base pointer, while sel_start and
 *     num_sel_chunks define the active subrange for the current batch.
 *     On exit, the original selection window is restored.
 *
 *   When H5SC_BATCH_INTO_SINGLE_CHUNK is defined, the function instead
 *     processes one selected chunk at a time for debugging. That path is
 *     intended for investigation and validation and should preserve its
 *     existing commented debugging code.
 *
 *   On failure, the function performs conservative pin unwinding for any
 *     selected chunks that were pinned during I/O setup but not yet
 *     unpinned by the write path, ensuring chunk_counter is decremented
 *     at most once for each remaining pinned selection entry.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state used for budgeting, eviction, and
 *     write execution.
 *
 *   H5D_dset_io_info_t *dset_info:
 *     Pointer to the dataset I/O context for the write being processed.
 *
 *   H5SC_dset_header_t *dset_hdr:
 *     Pointer to the SCC dataset header associated with this I/O request.
 *     Its io_info field supplies the selected-chunk window to batch.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__invoke_write_dset_batched(H5SC_t *cache, H5D_dset_io_info_t *dset_info, H5SC_dset_header_t *dset_hdr)
{
    H5SC_io_info_t      *io;
    H5SC_io_sel_chunk_t *saved_sel;
    size_t               saved_start;
    size_t               saved_n;
    const size_t        *saved_order;
    size_t               saved_resident_n;
    size_t               processed = 0;

    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset_info);
    assert(dset_hdr);
    io = dset_hdr->io_info;
    assert(io);

    saved_sel        = io->sel_chunks;
    saved_order      = io->sel_order;
    saved_start      = io->sel_start;
    saved_n          = io->num_sel_chunks;
    saved_resident_n = io->num_resident_sel_chunks;

    if (saved_n == 0)
        HGOTO_DONE(SUCCEED);

#ifdef H5SC_BATCH_INTO_SINGLE_CHUNK
    /* Process each selected chunk individually (debugging mode). */
    size_t i = 0;
    for (i = 0; i < saved_n; i++) {
        io->sel_chunks     = saved_sel;
        io->sel_order      = saved_order ? &saved_order[i] : NULL;
        io->sel_start      = saved_order ? saved_start : saved_start + i;
        io->num_sel_chunks = 1;

#ifdef DO_SANITY_CHECKS
        H5SC_test_write_batch_calls++;
        H5SC_test_write_last_batch_len = io->num_sel_chunks;
        if (io->num_sel_chunks > H5SC_test_write_max_batch_len)
            H5SC_test_write_max_batch_len = io->num_sel_chunks;
#endif

        if (H5SC_write(cache, dset_info) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to process batched SCC write");
        }

        H5SC__stats_record_batch_write(cache);
        H5SC__stats_record_batch_write_len(cache, 1);
    }
    goto done;
#endif

    /*
     * Full batching:
     *  - Build the largest batch that fits within the current budget
     *  - Evict before dispatching the batch
     *  - Repeat until the full selection window is processed
     */

    while (processed < saved_n) {
        size_t budget;
        size_t phase_end;

        if (processed < saved_resident_n)
            phase_end = saved_resident_n;
        else
            phase_end = saved_n;

        budget = H5SC__effective_budget(cache);

        /* Configure io_info to expose this provisional window. */
        io->sel_order      = saved_order ? &saved_order[processed] : NULL;
        io->sel_start      = saved_order ? saved_start : saved_start + processed;
        io->num_sel_chunks = phase_end - processed;

        /* Populate lookup metadata for the provisional window. */
        if (H5SC__lookup_cache_misses(dset_info->dset, io) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTGET, FAIL, "invoke write: miss lookup caching failed");

        size_t batch_len       = 0;
        size_t batch_need      = 0;
        bool   oversized_batch = false;

        /* Greedily extend batch while staying within budget */
        for (size_t j = 0; j < io->num_sel_chunks; j++) {
            H5SC_io_sel_chunk_t *sel     = H5SC__io_sel_at(io, j);
            H5SC_chunk_t        *chk     = sel->cached_chunk;
            size_t               desired = 0;
            size_t               inc     = 0;

            assert(chk);

            /*
             * Estimate the final resident allocation for a nonresident chunk. The
             * estimator currently considers the lookup allocation hint, dataset-local
             * observed history, and the dense logical size as a final fallback.
             *
             * Conservative write-growth allowance is intentionally disabled during the
             * initial statistics-gathering phase.
             */
            if (!chk->chunk_obj) {
                H5SC_size_est_source_t source;

                if (H5SC__estimate_resident_size(dset_hdr, dset_info->dset, sel, true, 0, &source, &desired) <
                    0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANTGETSIZE, FAIL,
                                "invoke write: unable to estimate resident chunk size");

                sel->estimate_resident_size = desired;
                sel->estimate_source        = source;
                sel->estimate_pending       = true;

                /*
                 * IMPORTANT: keep budget semantics identical to the core SCC code.
                 * This handles placeholder/shell accounting correctly.
                 */
                inc = H5SC__calc_inc_for_chunk(dset_info->dset, chk, desired);
            }

            if (inc > budget) {
                /*
                 * Dispatch any ordinary batch already under construction. This
                 * candidate will become the first candidate in the next iteration and
                 * will then be admitted as an oversized one-chunk batch.
                 */
                if (batch_len > 0)
                    break;

                /*
                 * The selection cannot be divided below this one chunk. Admit it as a
                 * one-entry oversized batch after reclaiming everything currently
                 * reclaimable under the active policy.
                 */
                batch_need      = inc;
                batch_len       = 1;
                oversized_batch = true;
                break;
            }

            if (batch_need > budget || inc > budget - batch_need) {
                break;
            }

            batch_need += inc;
            batch_len++;
        }

        assert(batch_len > 0);

        /* Evict before dispatching the batch. batch_need is incremental only. */
        if (batch_need > 0) {
            if (H5SC__ensure_space(cache, batch_need, oversized_batch) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                            "invoke write: unable to prepare SCC space for batch");
        }

        if (oversized_batch)
            H5SC__report_oversized_admission(cache, batch_need);

        /* Finalize the indexed active window for this batch. */
        io->sel_chunks     = saved_sel;
        io->sel_order      = saved_order ? &saved_order[processed] : NULL;
        io->sel_start      = saved_order ? saved_start : saved_start + processed;
        io->num_sel_chunks = batch_len;

#ifdef DO_SANITY_CHECKS
        H5SC_test_write_batch_calls++;
        H5SC_test_write_last_batch_len = io->num_sel_chunks;
        if (io->num_sel_chunks > H5SC_test_write_max_batch_len)
            H5SC_test_write_max_batch_len = io->num_sel_chunks;
#endif

        if (H5SC_write(cache, dset_info) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "invoke write: H5SC_write failed");

        if (oversized_batch) {
            /*
             * This can flush and evict the just-written dirty chunk. That is
             * intentional: a chunk too large for the active limit must not remain
             * resident and supply artificial budget to later batches.
             */
            if (H5SC__ensure_space(cache, 0, false) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                            "invoke write: unable to restore configured active limit "
                            "after oversized chunk");
        }

        H5SC__stats_record_batch_write(cache);
        H5SC__stats_record_batch_write_len(cache, batch_len);

        processed += batch_len;
    }

done:
    /*
     * Restore the complete request window before inspecting or unwinding
     * selection-owned pins. This makes H5SC__io_sel_at() resolve indices
     * against the original selection and ordering arrays rather than the
     * final active batch.
     */
    if (dset_hdr && dset_hdr->io_info) {
        dset_hdr->io_info->sel_chunks              = saved_sel;
        dset_hdr->io_info->sel_order               = saved_order;
        dset_hdr->io_info->sel_start               = saved_start;
        dset_hdr->io_info->num_sel_chunks          = saved_n;
        dset_hdr->io_info->num_resident_sel_chunks = saved_resident_n;

        /*
         * A successful batched write must have released every
         * selection-owned pin. Report an ownership leak instead of silently
         * correcting it.
         */
        if (ret_value >= 0) {
            for (size_t j = 0; j < saved_n; j++) {
                H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(dset_hdr->io_info, j);

                if (sel->pin_held) {
                    HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "batched write completed with an owned pin");
                    break;
                }
            }
        }

        /*
         * Unwind pins only after the operation has failed. This includes a
         * successful processing path converted to failure by the ownership
         * check above.
         */
        if (ret_value < 0) {
            for (size_t j = 0; j < saved_n; j++) {
                H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(dset_hdr->io_info, j);
                H5SC_chunk_t        *chk = sel->cached_chunk;

                if (sel->pin_held) {
                    if (!chk || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                        HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                    "batched write cleanup encountered an invalid owned pin");
                    else {
                        sel->pin_held = false;
                        chk->last_op  = H5SC_TAG_UNPIN_WRITE_DONE_ERROR;
                    }
                }
            }
        }
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__invoke_write_dset_batched()*/

/*-------------------------------------------------------------------------
 * Function: H5SC_invoke_write
 *
 * Purpose:
 *   Function that initiates one or more dataset write operations through
 *     the SCC. The function first initializes per-request I/O selection
 *     state via H5SC__io_info_init(), then processes each dataset
 *     independently through the batched SCC write path.
 *
 *   For each dataset in the request, the function locates the associated
 *     SCC dataset header, records the live dataset pointer needed by
 *     callback-driven SCC operations, executes the batched write path,
 *     promotes the dataset header to the head of the global dataset LRU,
 *     resets active accounting, and trims retained cache state back to the
 *     configured quiescent limit. Before returning, the function resets the
 *     per-dataset io_info state created for the request.
 *
 *   This function does not itself flush all cached chunks; any flushing is
 *     performed only as required by lower-level SCC eviction or trim
 *     behavior.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state used for request processing.
 *
 *   size_t count:
 *     Number of dataset I/O entries in dset_info.
 *
 *   H5D_dset_io_info_t *dset_info:
 *     Pointer to the array of dataset I/O request descriptors to process.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

herr_t
H5SC_invoke_write(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info)
{
    herr_t ret_value = SUCCEED;
    bool   io_inited = false;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    assert(count != (size_t)0);
    assert(dset_info);

    if (H5SC__io_info_init(cache, count, dset_info) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_INIT_CHUNK_SELECTION, FAIL,
                    "failed to finish the io_info_init call (SCC write)");
    }
    io_inited = true;

    /* Process each dataset independently, splitting into batches if necessary */
    for (size_t i = 0; i < count; i++) {
        H5SC_dset_header_t *dset_hdr = H5SC__ht_dset_find(cache, dset_info[i].dset->oloc.addr);
        if (!dset_hdr) {
            HGOTO_ERROR(H5E_SCC, H5E_NOT_A_MEMBER, FAIL, "dataset header not found for SCC write");
        }

        /* Record the live dataset pointer for eviction/flush callbacks */
        dset_hdr->dset = dset_info[i].dset;

        if (H5SC__invoke_write_dset_batched(cache, &dset_info[i], dset_hdr) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to process batched SCC write");
        }
    }

    /* After the operation completes, promote each involved dataset to head of the dset LRU list */
    for (size_t i = 0; i < count; i++) {
        H5SC_dset_header_t *dset_hdr = H5SC__ht_dset_find(cache, dset_info[i].dset->oloc.addr);
        if (dset_hdr && H5SC__dset_lru_promote(cache, dset_hdr) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL,
                        "failed to promote dataset header in LRU (SCC write)");
        }
    }

    /* End-of-request enforcement: reset active accounting and trim retained cache */
    cache->SCC_active_size = 0;
    if (H5SC__trim_to_quiescent_limit(cache) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed to trim cache to quiescent limit (write)");
    }

done:
    /* Reset per-dataset io_info state created for this request. */
    if (io_inited) {
        for (size_t i = 0; i < count; i++) {
            H5SC_dset_header_t *dset_hdr = H5SC__ht_dset_find(cache, dset_info[i].dset->oloc.addr);
            if (dset_hdr && dset_hdr->io_info) {
                if (H5SC__io_info_reset(dset_hdr->io_info) < 0) {
                    HDONE_ERROR(H5E_SCC, H5E_CANTRELEASE, FAIL, "failed to reset dataset io_info (write)");
                    ret_value = FAIL;
                }
            }
        }
    }
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_invoke_write() */

/*-------------------------------------------------------------------------
 * Function: H5SC_invoke_read
 *
 * Purpose:
 *   Function that initiates one or more dataset read operations through
 *     the SCC. The function first initializes per-request I/O selection
 *     state via H5SC__io_info_init(), then processes each dataset
 *     independently through the batched SCC read path.
 *
 *   For each dataset in the request, the function locates the associated
 *     SCC dataset header, records the live dataset pointer needed by
 *     callback-driven SCC operations, executes the batched read path,
 *     promotes the dataset header to the head of the global dataset LRU,
 *     resets active accounting, and trims retained cache state back to the
 *     configured quiescent limit. Before returning, the function resets the
 *     per-dataset io_info state created for the request.
 *
 *   This function does not itself flush cached chunks except indirectly if
 *     lower-level SCC trimming or eviction logic requires it.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state used for request processing.
 *
 *   size_t count:
 *     Number of dataset I/O entries in dset_info.
 *
 *   H5D_dset_io_info_t *dset_info:
 *     Pointer to the array of dataset I/O request descriptors to process.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

herr_t
H5SC_invoke_read(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info)
{
    herr_t ret_value = SUCCEED;
    bool   io_inited = false;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    assert(count != (size_t)0);
    assert(dset_info);

    if (H5SC__io_info_init(cache, count, dset_info) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_INIT_CHUNK_SELECTION, FAIL,
                    "failed to finish the io_info_init call (SCC read)");
    }

    io_inited = true;

    /* Process each dataset independently, splitting into batches if necessary */
    for (size_t i = 0; i < count; i++) {
        H5SC_dset_header_t *dset_hdr = H5SC__ht_dset_find(cache, dset_info[i].dset->oloc.addr);
        if (!dset_hdr) {
            HGOTO_ERROR(H5E_SCC, H5E_NOT_A_MEMBER, FAIL, "dataset header not found for SCC read");
        }

        /* Record the live dataset pointer for eviction/flush callbacks */
        dset_hdr->dset = dset_info[i].dset;

        if (H5SC__invoke_read_dset_batched(cache, &dset_info[i], dset_hdr) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to process batched SCC read");
        }
    }

    /* After the operation completes, promote each involved dataset to head of the dset LRU list */
    for (size_t i = 0; i < count; i++) {
        H5SC_dset_header_t *dset_hdr = H5SC__ht_dset_find(cache, dset_info[i].dset->oloc.addr);
        if (dset_hdr && H5SC__dset_lru_promote(cache, dset_hdr) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL,
                        "failed to promote dataset header in LRU (SCC read)");
        }
    }

    /* Enforce the quiescent retention limit at the end of the request */
    cache->SCC_active_size = 0;
    if (H5SC__trim_to_quiescent_limit(cache) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed to trim SCC to quiescent limit (read)");
    }

done:
    /* Reset per-dataset io_info state created for this request. */
    if (io_inited) {
        for (size_t i = 0; i < count; i++) {
            H5SC_dset_header_t *dset_hdr = H5SC__ht_dset_find(cache, dset_info[i].dset->oloc.addr);
            if (dset_hdr && dset_hdr->io_info) {
                if (H5SC__io_info_reset(dset_hdr->io_info) < 0) {
                    HDONE_ERROR(H5E_SCC, H5E_CANTRELEASE, FAIL, "failed to reset dataset io_info (read)");
                    ret_value = FAIL;
                }
            }
        }
    }
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_invoke_read() */

/*-------------------------------------------------------------------------
 * Function: H5SC__lookup_cache_misses
 *
 * Purpose:
 *   Function that performs layout-level lookup operations for the subset of
 *     selected chunks in the active I/O window that are not currently
 *     resident and do not already have invoke-local lookup metadata cached.
 *
 *   The function follows the current selection-window semantics of
 *     H5SC_read() and H5SC_write(): sel_chunks remains the stable base
 *     pointer for the selection array, while sel_start and num_sel_chunks
 *     define the active subrange to examine. For each eligible chunk in
 *     that window, the function batches the lookup inputs, invokes the
 *     layout client's lookup callback once, and stores the returned lookup
 *     results in the per-selection lookup_* fields.
 *
 *   In addition to caching per-selection lookup metadata for the current
 *     invoke, the function attaches returned udata to the corresponding
 *     cached chunk if that chunk does not already have udata assigned.
 *
 *   Compact callback argument arrays are obtained from the reusable
 *     H5SC_io_scratch_t workspace rather than allocated per invocation.
 *     Only the first miss_count entries are valid for the lookup callback;
 *     the workspace contents are transient and may be overwritten by later
 *     SCC processing in the same request.
 *
 * Inputs:
 *   H5D_t *dset:
 *     Pointer to the dataset whose layout lookup callback is to be used.
 *
 *   H5SC_io_info_t *sc_io_info:
 *     Pointer to the SCC I/O state describing the active selected-chunk
 *     window to process.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__lookup_cache_misses(H5D_t *dset, H5SC_io_info_t *sc_io_info)
{
    size_t          miss_count    = 0;
    haddr_t         md_tag        = HADDR_UNDEF;
    bool            md_tag_set    = false;
    herr_t          ret_value     = SUCCEED;
    const hsize_t **scaled_miss   = NULL;
    haddr_t       **addr_miss     = NULL;
    hsize_t       **size_miss     = NULL;
    hsize_t       **def_sz_miss   = NULL;
    size_t        **hint_miss     = NULL;
    size_t        **def_hint_miss = NULL;
    void          **udata_miss    = NULL;
    size_t         *miss_idx      = NULL;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(sc_io_info);
    assert(dset->shared->layout.sc_ops->lookup);

    /* Count non-resident selections that still need lookup metadata. */
    for (size_t j = 0; j < sc_io_info->num_sel_chunks; j++) {

        H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, j);
        H5SC_chunk_t        *chk = sel->cached_chunk;
        assert(chk);

        if (chk->chunk_obj)
            continue; /* resident => no lookup needed */

        if (sel->lookup_valid)
            continue; /* already cached for this invoke */

        miss_count++;
    }

    if (miss_count == 0) {
        HGOTO_DONE(SUCCEED);
    }

    /*
     * Ensure the reusable compact lookup workspace can represent all misses in
     * the current active selection window.
     */
    if (H5SC__io_scratch_ensure(sc_io_info, sc_io_info->num_sel_chunks) < 0)
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to prepare SCC compact lookup scratch storage");

    assert(sc_io_info->scratch);
    assert(sc_io_info->scratch->alloced >= miss_count);

    scaled_miss   = sc_io_info->scratch->lookup_scaled;
    addr_miss     = sc_io_info->scratch->lookup_addr;
    size_miss     = sc_io_info->scratch->lookup_size;
    def_sz_miss   = sc_io_info->scratch->lookup_defined_values_size;
    hint_miss     = sc_io_info->scratch->lookup_size_hint;
    def_hint_miss = sc_io_info->scratch->lookup_defined_values_size_hint;
    udata_miss    = sc_io_info->scratch->lookup_udata;
    miss_idx      = sc_io_info->scratch->lookup_idx;

    /* Build lookup argument arrays for the current miss set. */
    {
        size_t k = 0;

        for (size_t j = 0; j < sc_io_info->num_sel_chunks; j++) {
            H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, j);
            H5SC_chunk_t        *chk = sel->cached_chunk;

            if (chk->chunk_obj || sel->lookup_valid == true)
                continue;

            miss_idx[k]    = j;
            scaled_miss[k] = sel->scaled;

            addr_miss[k]     = &sel->lookup_addr;
            size_miss[k]     = &sel->lookup_disk_nbytes;
            def_sz_miss[k]   = &sel->lookup_defined_values_size;
            hint_miss[k]     = &sel->lookup_size_hint;
            def_hint_miss[k] = &sel->lookup_defined_values_size_hint;

            /*
             * lookup_udata is output storage for this callback invocation. Do not
             * retain a pointer left over from an earlier compact lookup.
             */
            udata_miss[k] = NULL;

            k++;
        }

        assert(k == miss_count);

        /* Tag metadata accesses for the batched lookup. */
        H5AC_tag(dset->oloc.addr, &md_tag);
        md_tag_set = true;

        if (dset->shared->layout.sc_ops->lookup(dset, miss_count, scaled_miss, addr_miss, size_miss,
                                                def_sz_miss, hint_miss, def_hint_miss, udata_miss) < 0) {
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to lookup chunk (SCC)");
        }

        /* Commit returned lookup metadata. */
        for (size_t k2 = 0; k2 < miss_count; k2++) {
            size_t               j   = miss_idx[k2];
            H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, j);
            H5SC_chunk_t        *chk = sel->cached_chunk;

#if H5SC_DO_SANITY_CHECKS
            assert(!sel->lookup_udata);
            assert(!sel->lookup_valid);
#endif

            sel->lookup_udata = udata_miss[k2];
            sel->lookup_valid = true;

            assert(chk);

#if H5SC_DO_SANITY_CHECKS
            assert(!chk->chunk_obj);
            assert(!chk->udata);
#endif

            chk->udata = udata_miss[k2];
        }
    }

done:
    if (md_tag_set)
        H5AC_tag(md_tag, NULL); /* Reset the metadata tag for the next dataset */

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__lookup_cache_misses() */

/*-------------------------------------------------------------------------
 * Function: H5SC__io_sel_chunk_init
 *
 * Purpose:
 *   Function that initializes an H5SC_io_sel_chunk_t structure to a
 *     consistent default state for use in SCC I/O processing. The
 *     structure is zero-initialized and key sentinel values are set so
 *     that all fields have well-defined semantics prior to population.
 *
 *   This function is intended to be called before populating per-chunk
 *     I/O selection state, ensuring that pointer fields, lookup metadata,
 *     dataspace handles, and coordinate fields begin in a known and safe
 *     configuration.
 *
 * Inputs:
 *   H5SC_io_sel_chunk_t *chunk:
 *     Pointer to the chunk selection structure to initialize.
 *
 * Return:
 *   None.
 *-------------------------------------------------------------------------
 */

static inline void
H5SC__io_sel_chunk_init(H5SC_io_sel_chunk_t *chunk)
{
    /* Zero everything (safe defaults for most members) */
    memset(chunk, 0, sizeof(*chunk));

    /* Non-zero sentinel defaults */
    chunk->lookup_addr = HADDR_UNDEF;

    /* Everything else is now:
     * dset_info = NULL
     * cached_chunk = NULL
     * lookup_valid = false
     * lookup_* sizes = 0
     * file_space/mem_space = NULL
     * file_space_shared/mem_space_shared = false
     * buf.vp/cvp = NULL (via zeroing the flexible ptr union)
     * coords/scaled/chk_log_coord = 0
     */
}

static inline H5SC_io_sel_chunk_t *
H5SC__io_sel_at(const H5SC_io_info_t *io, size_t relative_idx)
{
    size_t absolute_idx;

    assert(io);
    assert(io->sel_chunks);
    assert(relative_idx < io->num_sel_chunks);

    if (io->sel_order)
        absolute_idx = io->sel_order[relative_idx];
    else
        absolute_idx = io->sel_start + relative_idx;

    assert(absolute_idx < io->sel_chunks_alloced);

    return &io->sel_chunks[absolute_idx];
} /* end H5SC__io_sel_at() */

static herr_t
H5SC__build_resident_first_order(H5SC_io_info_t *sc_io_info)
{
    size_t order_count = 0;
    herr_t ret_value   = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(sc_io_info);

    sc_io_info->sel_order               = NULL;
    sc_io_info->num_resident_sel_chunks = 0;

    if (sc_io_info->num_sel_chunks == 0)
        HGOTO_DONE(SUCCEED);

    if (H5SC__io_scratch_ensure(sc_io_info, sc_io_info->num_sel_chunks) < 0)
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate resident-first order vector");

    assert(sc_io_info->scratch);
    assert(sc_io_info->scratch->order_idx);
    assert(sc_io_info->scratch->alloced >= sc_io_info->num_sel_chunks);

    /*
     * First pass: resident decoded chunks, preserving their original order.
     */
    for (size_t i = 0; i < sc_io_info->num_sel_chunks; i++) {
        H5SC_io_sel_chunk_t *sel = &sc_io_info->sel_chunks[sc_io_info->sel_start + i];

        assert(sel->cached_chunk);

        if (sel->cached_chunk->chunk_obj)
            sc_io_info->scratch->order_idx[order_count++] = sc_io_info->sel_start + i;
    }

    sc_io_info->num_resident_sel_chunks = order_count;

    /*
     * Second pass: nonresident chunks and cached shells, preserving their
     * original order.
     */
    for (size_t i = 0; i < sc_io_info->num_sel_chunks; i++) {
        H5SC_io_sel_chunk_t *sel = &sc_io_info->sel_chunks[sc_io_info->sel_start + i];

        assert(sel->cached_chunk);

        if (!sel->cached_chunk->chunk_obj)
            sc_io_info->scratch->order_idx[order_count++] = sc_io_info->sel_start + i;
    }

    assert(order_count == sc_io_info->num_sel_chunks);

    sc_io_info->sel_order = sc_io_info->scratch->order_idx;

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

/*-------------------------------------------------------------------------
 * Function: H5SC__io_info_init
 *
 * Purpose:
 *   Function that initializes per-request SCC I/O state for one or more
 *     dataset I/O operations. For each dataset in the request, the function
 *     resets the dataset's H5SC_io_info_t structure, identifies the chunks
 *     intersecting the file selection, allocates or extends the selected-
 *     chunk array as needed, associates each selected chunk with its SCC
 *     cache entry, and computes the per-chunk file and memory dataspaces
 *     required for later SCC read/write processing.
 *
 *   For each selected chunk, the function computes the chunk's logical
 *     coordinate and SCC chunk key, looks up or creates the corresponding
 *     cached chunk entry, pins that chunk for the duration of request
 *     processing, and fills the H5SC_io_sel_chunk_t entry with coordinate,
 *     buffer, dataspace, and cache-reference information. The function
 *     supports both shape-same and projected-selection memory-space setup,
 *     and handles "all" selections by iterating chunk intersections across
 *     the dataset extent.
 *
 *   The cache parameter is included because chunk-cache state is consulted
 *     during initialization, including dataset-header lookup and cached-
 *     chunk lookup/creation.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state used for dataset-header lookup and
 *     cached-chunk lookup/creation during request initialization.
 *
 *   size_t count:
 *     Number of dataset I/O entries in dset_info.
 *
 *   H5D_dset_io_info_t *dset_info:
 *     Pointer to the array of dataset I/O request descriptors for which
 *     SCC per-request I/O state is to be initialized.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__io_info_init(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info)
{
    H5S_t          *tmp_dset_space     = NULL;
    H5S_t          *single_chunk_space = NULL;
    H5SC_io_info_t *sc_io_info         = NULL;
    size_t          i;
    herr_t          ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    assert(count == 0 || dset_info);

    /* Iterate over datasets */
    for (i = 0; i < count; i++) {
        H5S_sel_type file_sel_type;
        H5S_sel_type mem_sel_type;

        hsize_t  sel_points;
        hsize_t  chunk_dims[H5S_MAX_RANK];
        hsize_t  file_dims[H5S_MAX_RANK];
        hsize_t  mem_dims[H5S_MAX_RANK];
        unsigned file_ndims;
        unsigned mem_ndims;
        hsize_t  start_coords[H5O_LAYOUT_NDIMS]; /* Starting coordinates of selection */
        hsize_t  coords[H5S_MAX_RANK];           /* Current coordinates of chunk */
        hsize_t  end[H5S_MAX_RANK];              /* Final coordinates of chunk */
        hsize_t  start_scaled[H5S_MAX_RANK];     /* Starting scaled coordinates of selection */
        hsize_t  scaled[H5S_MAX_RANK];           /* Scaled coordinates for this chunk */
        hsize_t  file_sel_start[H5S_MAX_RANK];   /* Offset of low bound of file selection */
        hsize_t  file_sel_end[H5S_MAX_RANK];     /* Offset of high bound of file selection */
        unsigned num_partial_dims;
        hsize_t  curr_partial_clip[H5S_MAX_RANK];    /* Current partial dimension sizes to clip against */
        hsize_t  partial_dim_size[H5S_MAX_RANK];     /* Size of a partial dimension */
        bool is_partial_dim[H5S_MAX_RANK] = {false}; /* Whether a dimension is currently a partial chunk */
        int  curr_dim;                               /* Current dimension to increment */
        hssize_t adjust[H5S_MAX_RANK];               /* Adjustment to make to all file chunks (for shape same
                                                        algorithm)*/
        hsize_t zeros[H5S_MAX_RANK]; /* All zero vector (for start parameter to setting hyperslab on
                                        partial   chunks for "all" selection) */
        hsize_t  dset_sel_chunks;
        bool     shape_same;
        unsigned u;

        H5SC_dset_header_t *dset_hdr = H5SC__ht_dset_find(cache, dset_info[i].dset->oloc.addr);

#if H5SC_DO_SANITY_CHECKS
        /* Sanity check to ensure the is present in */
        if (dset_hdr == NULL) {
            {
                HGOTO_ERROR(H5E_SCC, H5E_HT_NOTFOUND, FAIL,
                            "dataset should have been added to the hash table upon creation.");
            }
        }
#endif

        /* Record the live dataset pointer for eviction teardown operations */
        dset_hdr->dset = dset_info[i].dset;

        /* Convenience pointer to the dataset's SCC I/O state. */
        sc_io_info = dset_hdr->io_info;
        assert(sc_io_info);
        if (H5SC__io_info_reset(sc_io_info) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to reset sc_io_info during an io init operation");
        }

        /* Get number of elements in the file selection */
        sel_points = H5S_GET_SELECT_NPOINTS(dset_info[i].file_space);

        /* Nothing to do if no points selected, I/O is skipped, or no shared chunk cache client */
        if (sel_points == 0 || dset_info[i].skip_io || !dset_info[i].dset->shared->layout.sc_ops)
            continue;

        dset_sel_chunks = 0;

        /* Get chunk dimensions */
        assert(dset_info[i].dset->shared->layout.sc_ops->layout_query);
        if (dset_info[i].dset->shared->layout.sc_ops->layout_query(dset_info[i].dset, chunk_dims, NULL,
                                                                   NULL) < 0) {
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query chunk dimensions");
        }

        /* Get dataspace ranks */
        file_ndims = (unsigned)H5S_GET_EXTENT_NDIMS(dset_info[i].file_space);
        mem_ndims  = (unsigned)H5S_GET_EXTENT_NDIMS(dset_info[i].mem_space);

        /* Get the file and memory selection types */
        if ((file_sel_type = H5S_GET_SELECT_TYPE(dset_info[i].file_space)) < H5S_SEL_NONE) {
            HGOTO_ERROR(H5E_DATASET, H5E_BADSELECT, FAIL, "unable to get type of selection");
        }
        if ((mem_sel_type = H5S_GET_SELECT_TYPE(dset_info[i].mem_space)) < H5S_SEL_NONE) {
            HGOTO_ERROR(H5E_DATASET, H5E_BADSELECT, FAIL, "unable to get type of selection");
        }

        /* Get file dataspace dimensions */
        if (H5S_get_simple_extent_dims(dset_info[i].file_space, file_dims, NULL) < 0) {
            assert(0);
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "can't get file dataspace dimensions");
        }

        /* Set up bounding box and initial chunk coordinates for chunk iteration. */

        /* Initialize num_partial_dims.  Only needed for all selection, but put it outside of that
         * block to stop compiler warnings */
        num_partial_dims = 0;

        /* Check for "all" selection */
        if (H5S_SEL_ALL == file_sel_type) {
            /* Set up partial chunk tracking and set the bounding box to the extent */
            memset(zeros, 0, sizeof(zeros));
            for (u = 0; u < file_ndims; u++) {
                /* Validate this chunk dimension */
                if (chunk_dims[u] == 0) {
                    HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "chunk size must be > 0, dim = %u ", u);
                }

                /* Set up start / end coordinates for first chunk */
                scaled[u] = start_scaled[u] = 0;
                coords[u] = start_coords[u] = 0;
                end[u]                      = chunk_dims[u] - 1;

                /* Set up selection bounds */
                file_sel_start[u] = 0;
                file_sel_end[u]   = file_dims[u] - 1;

                /* Initialize partial chunk dimension information */
                partial_dim_size[u] = file_dims[u] % chunk_dims[u];
                if (file_dims[u] < chunk_dims[u]) {
                    curr_partial_clip[u] = partial_dim_size[u];
                    is_partial_dim[u]    = true;
                    num_partial_dims++;
                }
                else {
                    curr_partial_clip[u] = chunk_dims[u];
                    is_partial_dim[u]    = false;
                }
            }

            /* Create "temporary" chunk for selection operations (copy file space) */
            if (NULL == (single_chunk_space = H5S_create_simple(file_ndims, chunk_dims, NULL))) {
                HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, FAIL, "unable to create dataspace for chunk");
            }
        }
        else {
            /* Get bounding box for file selection */
            if (H5S_SELECT_BOUNDS(dset_info[i].file_space, file_sel_start, file_sel_end) < 0) {
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "can't get file selection bound info");
            }

            /* Iterate over dimensions */
            for (u = 0; u < file_ndims; u++) {
                /* Validate this chunk dimension */
                if (chunk_dims[u] == 0) {
                    HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "chunk size must be > 0, dim = %u ", u);
                }

                /* Set initial chunk location & hyperslab size */
                scaled[u] = start_scaled[u] = file_sel_start[u] / chunk_dims[u];
                coords[u] = start_coords[u] = scaled[u] * chunk_dims[u];
                end[u]                      = (coords[u] + chunk_dims[u]) - 1;
            }
        }

        /* Determine whether file and memory selections are shape-compatible. */

        if ((shape_same = H5S_SELECT_SHAPE_SAME(dset_info[i].file_space, dset_info[i].mem_space)) == true) {
            hsize_t mem_sel_start[H5S_MAX_RANK]; /* Offset of low bound of memory selection */
            hsize_t mem_sel_end[H5S_MAX_RANK];   /* Offset of high bound of memory selection */

            /* The shapes are the same, compute the adjustment offset to use for the memory dataspace
             * calculation */

            /* H5D__read()/H5D__write() should have made sure the ranks are the same */
            assert(file_ndims == mem_ndims);

            /* Get bounding box for memory selection */
            if (H5S_SELECT_BOUNDS(dset_info[i].mem_space, mem_sel_start, mem_sel_end) < 0) {
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "can't get memory selection bound info");
            }

            for (u = 0; u < file_ndims; u++) {
                /* Calculate the adjustment for memory selection from file selection */
                H5_CHECK_OVERFLOW(file_sel_start[u], hsize_t, hssize_t);
                H5_CHECK_OVERFLOW(mem_sel_start[u], hsize_t, hssize_t);
                adjust[u] = (hssize_t)file_sel_start[u] - (hssize_t)mem_sel_start[u];
            }

            /* Get memory dataspace dimensions */
            if (H5S_get_simple_extent_dims(dset_info[i].mem_space, mem_dims, NULL) < 0) {
                assert(0);
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "can't get memory dataspace dimensions");
            }
        }
        else
            /* Create a dataspace with the same extent as the file dataspace */
            if (NULL == (tmp_dset_space = H5S_create_simple(file_ndims, file_dims, NULL))) {
                {
                    HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCREATE, FAIL, "unable to copy file dataspace");
                }
            }

        /*
         * Iterate through each chunk in the file selection's bounding box
         */

        while (sel_points) {
            /* Check for intersection of current chunk and file selection */
            if ((H5S_SEL_ALL == file_sel_type) ||
                (true == H5S_SELECT_INTERSECT_BLOCK(dset_info[i].file_space, coords, end))) {
                H5SC_io_sel_chunk_t *sel_chunk;

                /* Ensure there is room for another selected-chunk entry. */

                /* Check for no array allocated */
                if (!sc_io_info->sel_chunks) {
                    assert(sc_io_info->num_sel_chunks == 0);
                    assert(sc_io_info->sel_chunks_alloced == 0);

                    /* Allocate initial array */
                    if (NULL == (sc_io_info->sel_chunks = H5MM_calloc(H5SC_INIT_CHUNK_LIST_SIZE *
                                                                      sizeof(sc_io_info->sel_chunks[0]))))
                        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL,
                                    "can't allocate array of selected chunks");
                    sc_io_info->sel_chunks_alloced = H5SC_INIT_CHUNK_LIST_SIZE;
                }
                else if (sc_io_info->num_sel_chunks == sc_io_info->sel_chunks_alloced) {
                    /* Out of space, double array size */
                    void *tmp_ptr;
                    if (NULL == (tmp_ptr = H5MM_realloc(sc_io_info->sel_chunks,
                                                        2 * sc_io_info->sel_chunks_alloced *
                                                            sizeof(sc_io_info->sel_chunks[0])))) {
                        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL,
                                    "can't reallocate array of selected chunks");
                    }
                    sc_io_info->sel_chunks = tmp_ptr;
                    size_t old_alloc       = sc_io_info->sel_chunks_alloced;
                    sc_io_info->sel_chunks_alloced *= 2;

                    /* Set initial values for newly added sel_chunks elements. */
                    for (size_t j = old_alloc; j < sc_io_info->sel_chunks_alloced; j++) {
                        H5SC__io_sel_chunk_init(&sc_io_info->sel_chunks[j]);
                    }
                }
                assert(sc_io_info->num_sel_chunks < sc_io_info->sel_chunks_alloced);

                /* Set convenience pointer */
                sel_chunk = &(sc_io_info->sel_chunks[sc_io_info->num_sel_chunks]);

                H5SC__io_sel_chunk_init(sel_chunk);

                /* Compute the logical chunk index and SCC chunk key. */
                hsize_t log_chk_idx = 0;
                if (H5SC__compute_logical_chunk_index(file_ndims, file_dims, chunk_dims, coords,
                                                      &log_chk_idx) < 0) {
                    HGOTO_ERROR(H5E_SCC, H5E_CANTCOMPUTE, FAIL, "failed to compute the linear chunk index");
                }

                sel_chunk->chk_log_coord = log_chk_idx;

                H5SC_chunk_key_t chk_key;
                if (H5SC__compute_chunk_key(&(dset_info[i].dset->oloc.addr), &(sel_chunk->chk_log_coord),
                                            &chk_key) < 0) {
                    HGOTO_ERROR(H5E_SCC, H5E_CHUNK_KEY, FAIL,
                                "failed to compute chunk key during io info init");
                }

                H5SC_chunk_t *cached_chk = H5SC__ht_chunk_find(cache, &chk_key);

                if (cached_chk) {
                    cached_chk->chk_log_coord = log_chk_idx;
                    sel_chunk->cached_chunk   = cached_chk;

                    if (H5SC__chunk_pin(cache, dset_hdr, cached_chk) < 0)
                        HGOTO_ERROR(H5E_SCC, H5E_CANTPIN, FAIL,
                                    "unable to pin cached chunk during I/O initialization");

                    sel_chunk->pin_held = true;

                    sel_chunk->cached_chunk->last_op = H5SC_TAG_PIN_IOINIT_CACHED;
                }
                else {
                    cached_chk = H5SC__make_chunk(chk_key, (size_t)0, (size_t)0, false);

                    if (!cached_chk) {
                        HGOTO_ERROR(H5E_SCC, H5E_CANNOTALLOC, FAIL,
                                    "failed to create chunk during init process");
                    }

                    if (H5SC__ht_chunk_insert(cache, cached_chk) < 0) {
                        HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL,
                                    "failed to insert chunk into hash table during io init process");
                    }

                    if (H5SC__chunk_lru_prepend(cache, dset_hdr, cached_chk) < 0) {
                        HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL,
                                    "failed to insert chunk into dset dll during io init process");
                    }
                    /* Increase pin count to retain this chunk during mid-I/O request processing evictions
                     */
                    cached_chk->chk_log_coord = log_chk_idx;

                    if (H5SC__chunk_pin(cache, dset_hdr, cached_chk) < 0)
                        HGOTO_ERROR(H5E_SCC, H5E_CANTPIN, FAIL,
                                    "unable to pin new chunk during I/O initialization");

                    sel_chunk->pin_held = true;

                    cached_chk->last_op     = H5SC_TAG_PIN_IOINIT;
                    sel_chunk->cached_chunk = cached_chk;
                }

                /* Set up selected chunk struct */
                sel_chunk->dset_info = &dset_info[i];
                sel_chunk->buf       = dset_info[i].buf;
                H5MM_memcpy(sel_chunk->coords, coords, sizeof(hsize_t) * file_ndims);
                H5MM_memcpy(sel_chunk->scaled, scaled, sizeof(hsize_t) * file_ndims);
                sel_chunk->file_space        = NULL;
                sel_chunk->mem_space         = NULL;
                sel_chunk->file_space_shared = false;
                sel_chunk->mem_space_shared  = false;

                sc_io_info->num_sel_chunks++;

                /* Set up chunk file dataspace including selection */

                /* Different actions for different file selection types */
                if (H5S_SEL_ALL == file_sel_type) {
                    /* "all" selection in file, simply reuse single chunk dataspace, select valid
                     * elements if it's a partial edge chunk */
                    /* Set the file chunk dataspace */
                    if (NULL == (sel_chunk->file_space = H5S_copy(single_chunk_space, true, false)))
                        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy chunk dataspace");

                    /* If there are partial dimensions for this chunk, set the hyperslab for them
                     */
                    if (num_partial_dims > 0)
                        if (H5S_select_hyperslab(sel_chunk->file_space, H5S_SELECT_SET, zeros, NULL,
                                                 curr_partial_clip, NULL) < 0)
                            HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "can't create chunk selection");
                }
                else {
                    if (H5S_SEL_HYPERSLABS == file_sel_type) {
                        hsize_t clip_count[H5S_MAX_RANK];

                        /* Hyperslab selection in file, create dataspace for chunk, 'AND'ing the
                         * overall selection with the current chunk.
                         *
                         * Clip the chunk block to the actual dataset extent before combining.
                         * This is required for selections that span from a full chunk into a
                         * partial edge chunk. Otherwise, the partial edge chunk can appear to
                         * contain a full chunk's worth of selected points.
                         */
                        for (u = 0; u < file_ndims; u++) {
                            if (coords[u] >= file_dims[u])
                                HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                            "attempted to build selection for chunk outside dataset extent");

                            clip_count[u] = MIN(chunk_dims[u], file_dims[u] - coords[u]);

                            if (clip_count[u] == 0)
                                HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                            "computed zero-sized clipped chunk selection");
                        }

                        if (H5S_combine_hyperslab(dset_info[i].file_space, H5S_SELECT_AND, coords, NULL,
                                                  clip_count, NULL, &sel_chunk->file_space) < 0)
                            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL,
                                        "unable to combine file space selection with clipped chunk block");
                    }
                    else {

                        /* H5S_SEL_POINTS */
                        /* Iterate over the file selection to create a new selection using only
                         * the points that are in this chunk. This algorithm is probably less
                         * efficient than the one in H5Dchunk.c that iterates over the points once
                         * and adds chunks involved to a skip list.  We may want to change it to
                         * that in the future. */
                        HGOTO_ERROR(H5E_DATASPACE, H5E_UNSUPPORTED, FAIL,
                                    "point selections are not yet supported (SCC)");
                    }

                    /* Resize chunk's dataspace dimensions to size of chunk */
                    if (H5S_set_extent_real(sel_chunk->file_space, chunk_dims) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "can't adjust chunk dimensions");

                    /* Move selection back to have correct offset in chunk */
                    if (H5S_SELECT_ADJUST_U(sel_chunk->file_space, coords) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "can't adjust chunk selection");
                }

                /* Decrement the number of points left in this file selection. */
                hsize_t chunk_points = H5S_GET_SELECT_NPOINTS(sel_chunk->file_space);

                if (chunk_points == 0 || chunk_points > sel_points)
                    HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                "invalid selected chunk point count during SCC I/O init");

                sel_points -= chunk_points;

                /* Increment number of chunks selected in dataset */
                dset_sel_chunks++;

                /*
                 * Set up chunk memory dataspace including selection
                 */

                /* Check for only one chunk selected in this dataset */
                if (sel_points == 0 && dset_sel_chunks == 1) {
                    /* Since only one chunk is selected, no complicated transformation is
                     * necessary to get the matching memory space. Just point at the entire memory
                     * dataspace & selection */
                    sel_chunk->mem_space = dset_info[i].mem_space;

                    /* Indicate that the chunk's memory space is shared. In this case, the memory
                     * space is a of what was passed into the I/O call and will not need to be
                     * freed by H5SC__io_info_term() */
                    sel_chunk->mem_space_shared = true;
                }
                else {
                    if (shape_same) {
                        H5S_sel_type chunk_sel_type; /* Chunk's selection type */
                        /* Dataspace selections are the same shape in memory and the file, copy
                         * the file selection to memory and offset it as necessary to match */
                        /* Create chunk memory dataspace with the same extent as the overall
                         * memory dataspace
                         */
                        if ((sel_chunk->mem_space = H5S_create_simple(mem_ndims, mem_dims, NULL)) == NULL)
                            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCREATE, FAIL, "unable to create memory space");

                        /* Get the chunk's selection type */
                        if ((chunk_sel_type = H5S_GET_SELECT_TYPE(sel_chunk->file_space)) < H5S_SEL_NONE)
                            HGOTO_ERROR(H5E_DATASET, H5E_BADSELECT, FAIL, "unable to get type of selection");

                        /* Set memory selection */
                        if (H5S_SEL_ALL == chunk_sel_type) {
                            hsize_t mem_coords[H5S_MAX_RANK];

                            /* "all" selection in chunk simply select the entire chunk within the
                             * memory space, offset as necessary */

                            /* Adjust the chunk coordinates */
                            for (u = 0; u < file_ndims; u++)
                                mem_coords[u] = (hsize_t)((hssize_t)coords[u] - adjust[u]);

                            /* Set to same shape as chunk */
                            if (H5S_select_hyperslab(sel_chunk->mem_space, H5S_SELECT_SET, mem_coords, NULL,
                                                     chunk_dims, NULL) < 0)
                                HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL,
                                            "can't create chunk memory selection");
                        }
                        else {
                            hssize_t piece_adjust[H5S_MAX_RANK];

                            /* Hyperslab or point election, copy chunk file selection to memory
                             * and offset
                             */

                            /* Sanity check */
                            assert(H5S_SEL_HYPERSLABS == file_sel_type);

                            /* Copy the file chunk's selection */
                            if (H5S_SELECT_COPY(sel_chunk->mem_space, sel_chunk->file_space, false) < 0)
                                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy selection");

                            /* Compute the adjustment for this chunk */
                            for (u = 0; u < file_ndims; u++) {
                                /* Compensate for the chunk offset */
                                H5_CHECK_OVERFLOW(coords[u], hsize_t, hssize_t);
                                piece_adjust[u] = adjust[u] - (hssize_t)coords[u];
                            } /* end for */

                            /* Adjust the selection */
                            if (H5S_SELECT_ADJUST_S(sel_chunk->mem_space, piece_adjust) < 0)
                                HGOTO_ERROR(H5E_DATASET, H5E_CANTSET, FAIL, "unable to adjust selection");
                        }
                    }
                    else {
                        /* Create dataspace for entire current chunk within the file space.
                         * Shouldn't matter if it goes beyond the extent since* we're not doing
                         * I/O with this space
                         */
                        if (H5S_select_hyperslab(tmp_dset_space, H5S_SELECT_SET, coords, NULL, chunk_dims,
                                                 NULL) < 0)
                            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, FAIL, "unable to select chunk block");

                        /* Calculate memory selection for this chunk by projecting intersection of
                         * full file selection and file chunk to full memory selection. Note that
                         * we share the selection so we cannot further modify sel_chunk->mem_space
                         * (it can be closed).
                         */
                        if (H5S_select_project_intersection(dset_info[i].file_space, dset_info[i].mem_space,
                                                            tmp_dset_space, &sel_chunk->mem_space, true) < 0)
                            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL, "can't project intersection");
                    }
                }
            }

            /* Advance to next chunk within bounding box */

            /* Set current increment dimension */
            curr_dim = (int)file_ndims - 1;

            /* Increment chunk location in fastest changing dimension */
            coords[curr_dim] += chunk_dims[curr_dim];
            end[curr_dim] += chunk_dims[curr_dim];
            scaled[curr_dim]++;

            /* Bring chunk location back into bounds, if necessary */
            if (coords[curr_dim] > file_sel_end[curr_dim]) {
                do {
                    /* Reset current dimension's location to 0 */
                    scaled[curr_dim] = start_scaled[curr_dim];
                    coords[curr_dim] = start_coords[curr_dim];
                    end[curr_dim]    = (coords[curr_dim] + chunk_dims[curr_dim]) - 1;

                    /* Check for previous partial chunk in this dimension */
                    if (is_partial_dim[curr_dim] && end[curr_dim] < file_dims[curr_dim]) {
                        /* Sanity checks */
                        assert(num_partial_dims > 0);
                        assert(H5S_SEL_ALL == file_sel_type);

                        /* Reset partial chunk information for this dimension */
                        curr_partial_clip[curr_dim] = chunk_dims[curr_dim];
                        is_partial_dim[curr_dim]    = false;
                        num_partial_dims--;
                    } /* end if */

                    /* Decrement current dimension */
                    curr_dim--;

                    /* Check for valid current dim */
                    if (curr_dim >= 0) {
                        /* Increment chunk location in current dimension */
                        scaled[curr_dim]++;
                        coords[curr_dim] += chunk_dims[curr_dim];
                        end[curr_dim] = (coords[curr_dim] + chunk_dims[curr_dim]) - 1;
                    } /* end if */
                } while (curr_dim >= 0 && (coords[curr_dim] > file_sel_end[curr_dim]));

                /* Check for new partial chunk in this dimension for "all" selection. First check
                 * for valid current dim */
                if ((H5S_SEL_ALL == file_sel_type) && curr_dim >= 0) {
                    /* Check for partial chunk in this dimension */
                    if (!is_partial_dim[curr_dim] && file_dims[curr_dim] <= end[curr_dim]) {
                        /* Set partial chunk information for this dimension */
                        curr_partial_clip[curr_dim] = partial_dim_size[curr_dim];
                        is_partial_dim[curr_dim]    = true;
                        num_partial_dims++;

                        /* Sanity check */
                        assert(num_partial_dims <= file_ndims);
                    } /* end if */
                }     /* end if */
            }
        }

        /* Close temporary dataspaces */
        if (tmp_dset_space && H5S_close(tmp_dset_space) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary file dataspace");

        tmp_dset_space = NULL;

        if (single_chunk_space && H5S_close(single_chunk_space) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary file dataspace");

        single_chunk_space = NULL;

        if (H5SC__build_resident_first_order(sc_io_info) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTINIT, FAIL, "unable to build resident-first selection order");
    }

#if H5SC_DO_SANITY_CHECKS
    if (H5SC__verify_cached_reclaimability(cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                    "cached reclaimability mismatch after I/O initialization");
#endif

done:
    /* Clean up on failure */
    if (ret_value < 0) {
        /* Close temporary dataspaces */
        if (tmp_dset_space && H5S_close(tmp_dset_space) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary file dataspace");
        if (single_chunk_space && H5S_close(single_chunk_space) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary file dataspace");

        /* Terminate io info */
        if (sc_io_info) {
            if (H5SC__io_info_reset(sc_io_info) < 0) {
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't close I/O info");
            }
        }
    }
    else {
        assert(!tmp_dset_space);
        assert(!single_chunk_space);
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__io_info_init() */

/*-------------------------------------------------------------------------
 * Function: H5SC__selection_io_info_init
 *
 * Purpose:
 *   Initialize the selection-decomposition portion of an SCC request from
 *     a dataset file-space selection.
 *
 *   The function decomposes file_space into the logical structured chunks
 *     that intersect the selection. For each participating chunk, an
 *     H5SC_io_sel_chunk_t entry is populated with:
 *
 *       - the chunk origin in dataset coordinates;
 *       - the scaled chunk coordinates;
 *       - the dataset-relative logical chunk index; and
 *       - a chunk-local dataspace containing the portion of file_space
 *         that intersects that logical chunk.
 *
 *   Partial-edge chunks are clipped to the current dataset extent before
 *     the chunk-local selection is constructed. The resulting file_space
 *     stored in each selected-chunk entry is therefore expressed entirely
 *     in logical chunk coordinates and contains only valid in-extent
 *     elements.
 *
 *   This helper is intentionally cache-neutral. It does not perform SCC
 *     hash-table lookups, create H5SC_chunk_t entries, modify either SCC
 *     LRU, pin chunks, perform on-disk chunk lookups, or alter cache
 *     accounting. Cache-specific behavior is left to the caller.
 *
 *   The helper is shared by operations that require the same file-selection
 *     decomposition but have different cache semantics. In particular,
 *     H5SC__erase_io_info_init() adds cache lookup/creation and pinning after
 *     this function returns, while H5SC_get_defined() uses the decomposed
 *     selections without modifying SCC residency.
 *
 *   H5S_SEL_ALL and H5S_SEL_HYPERSLABS selections are supported.
 *     H5S_SEL_POINTS selections are intentionally unsupported by the current
 *     SCC interface and are rejected.
 *
 * Inputs:
 *   H5D_t *dset:
 *     Pointer to the structured-chunk dataset associated with file_space.
 *     The dataset layout client's layout_query callback is used to obtain
 *     the logical chunk dimensions.
 *
 *   const H5S_t *file_space:
 *     Dataset dataspace containing the selection to decompose.
 *
 *   H5SC_io_info_t *sc_io_info:
 *     Dataset-owned SCC request state to populate. Existing transient
 *     selection state is reset before decomposition begins. The sel_chunks
 *     backing allocation may be retained and reused.
 *
 *     On success, num_sel_chunks identifies the number of valid
 *     H5SC_io_sel_chunk_t entries beginning at sel_chunks[0], and sel_start
 *     is set to zero.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__selection_io_info_init(H5D_t *dset, const H5S_t *file_space, H5SC_io_info_t *sc_io_info)
{
    H5S_t  *single_chunk_space = NULL;
    hsize_t chunk_dims[H5S_MAX_RANK];
    hsize_t file_dims[H5S_MAX_RANK];
    hsize_t start_coords[H5O_LAYOUT_NDIMS];
    hsize_t coords[H5S_MAX_RANK];
    hsize_t end[H5S_MAX_RANK];
    hsize_t start_scaled[H5S_MAX_RANK];
    hsize_t scaled[H5S_MAX_RANK];
    hsize_t file_sel_start[H5S_MAX_RANK];
    hsize_t file_sel_end[H5S_MAX_RANK];
    hsize_t zeros[H5S_MAX_RANK];

    hsize_t      sel_points;
    unsigned     file_ndims;
    H5S_sel_type file_sel_type;
    int          curr_dim;
    unsigned     u;
    herr_t       ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(file_space);
    assert(sc_io_info);
    assert(dset->shared->layout.sc_ops);
    assert(dset->shared->layout.sc_ops->layout_query);

    /*
     * Release transient state from the previous use of this structure.
     * The new selection is always constructed beginning at sel_chunks[0].
     */
    if (H5SC__io_info_reset(sc_io_info) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to reset SCC selection information");

    sc_io_info->sel_start = 0;

    /* Get number of elements in the file selection. */
    sel_points = H5S_GET_SELECT_NPOINTS(file_space);

    /* Nothing to decompose for an empty selection. */
    if (sel_points == 0)
        HGOTO_DONE(SUCCEED);

    /* Get chunk dimensions from the layout client. */
    if (dset->shared->layout.sc_ops->layout_query(dset, chunk_dims, NULL, NULL) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query chunk dimensions");

    /* Get dataspace rank. */
    file_ndims = (unsigned)H5S_GET_EXTENT_NDIMS(file_space);

    /* Get selection type. */
    if ((file_sel_type = H5S_GET_SELECT_TYPE(file_space)) < H5S_SEL_NONE)
        HGOTO_ERROR(H5E_DATASET, H5E_BADSELECT, FAIL, "unable to get type of selection");

    /*
     * Point selections are intentionally unsupported for the current SCC
     * I/O interface.
     */
    if (H5S_SEL_POINTS == file_sel_type)
        HGOTO_ERROR(H5E_DATASPACE, H5E_UNSUPPORTED, FAIL, "point selections are not yet supported by SCC");

    if (H5S_SEL_ALL != file_sel_type && H5S_SEL_HYPERSLABS != file_sel_type)
        HGOTO_ERROR(H5E_DATASPACE, H5E_UNSUPPORTED, FAIL, "unsupported file selection type for SCC");

    /* Get dataset dimensions. */
    if (H5S_get_simple_extent_dims(file_space, file_dims, NULL) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "can't get file dataspace dimensions");

    /*
     * Set up the bounding box and initial chunk coordinates used to
     * enumerate the selected chunks.
     */
    if (H5S_SEL_ALL == file_sel_type) {
        memset(zeros, 0, sizeof(zeros));

        for (u = 0; u < file_ndims; u++) {
            if (chunk_dims[u] == 0)
                HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "chunk size must be > 0, dim = %u", u);

            scaled[u] = start_scaled[u] = 0;
            coords[u] = start_coords[u] = 0;
            end[u]                      = chunk_dims[u] - 1;

            file_sel_start[u] = 0;
            file_sel_end[u]   = file_dims[u] - 1;
        }

        if (NULL == (single_chunk_space = H5S_create_simple(file_ndims, chunk_dims, NULL)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, FAIL, "unable to create dataspace for chunk");
    }
    else {
        assert(H5S_SEL_HYPERSLABS == file_sel_type);

        if (H5S_SELECT_BOUNDS(file_space, file_sel_start, file_sel_end) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "can't get file selection bound info");

        for (u = 0; u < file_ndims; u++) {
            if (chunk_dims[u] == 0)
                HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, FAIL, "chunk size must be > 0, dim = %u", u);

            scaled[u] = start_scaled[u] = file_sel_start[u] / chunk_dims[u];
            coords[u] = start_coords[u] = scaled[u] * chunk_dims[u];
            end[u]                      = (coords[u] + chunk_dims[u]) - 1;
        }
    }

    /*
     * Walk all chunks within the file-selection bounding box. A chunk is
     * added to sel_chunks only if it intersects the actual selection.
     */
    while (sel_points) {
        /*
         * H5S_SELECT_INTERSECT_BLOCK() is logically a selection query, but its
         * internal interface does not currently accept a const dataspace. Keep the
         * SCC selection interfaces const-correct; resolving the H5S signature is
         * outside the scope of this module.
         */
        if ((H5S_SEL_ALL == file_sel_type) ||
            (true == H5S_SELECT_INTERSECT_BLOCK((H5S_t *)file_space, coords, end))) {

            H5SC_io_sel_chunk_t *sel_chunk   = NULL;
            hsize_t              log_chk_idx = 0;

            /*
             * Allocate or extend the selected-chunk array as required.
             */
            if (!sc_io_info->sel_chunks) {
                assert(sc_io_info->num_sel_chunks == 0);
                assert(sc_io_info->sel_chunks_alloced == 0);

                if (NULL == (sc_io_info->sel_chunks =
                                 H5MM_calloc(H5SC_INIT_CHUNK_LIST_SIZE * sizeof(sc_io_info->sel_chunks[0]))))
                    HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "can't allocate array of selected chunks");

                sc_io_info->sel_chunks_alloced = H5SC_INIT_CHUNK_LIST_SIZE;
            }
            else if (sc_io_info->num_sel_chunks == sc_io_info->sel_chunks_alloced) {
                void  *tmp_ptr = NULL;
                size_t old_alloc;

                if (NULL ==
                    (tmp_ptr = H5MM_realloc(sc_io_info->sel_chunks, 2 * sc_io_info->sel_chunks_alloced *
                                                                        sizeof(sc_io_info->sel_chunks[0]))))
                    HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL,
                                "can't reallocate array of selected chunks");

                sc_io_info->sel_chunks = tmp_ptr;
                old_alloc              = sc_io_info->sel_chunks_alloced;
                sc_io_info->sel_chunks_alloced *= 2;

                for (size_t j = old_alloc; j < sc_io_info->sel_chunks_alloced; j++)
                    H5SC__io_sel_chunk_init(&sc_io_info->sel_chunks[j]);
            }

            assert(sc_io_info->num_sel_chunks < sc_io_info->sel_chunks_alloced);

            sel_chunk = &sc_io_info->sel_chunks[sc_io_info->num_sel_chunks];

            H5SC__io_sel_chunk_init(sel_chunk);

            /*
             * Preserve both dataset-coordinate and scaled chunk positions.
             * These fields are intentionally populated independently of SCC
             * cache state.
             */
            H5MM_memcpy(sel_chunk->coords, coords, sizeof(hsize_t) * file_ndims);
            H5MM_memcpy(sel_chunk->scaled, scaled, sizeof(hsize_t) * file_ndims);

            /* Compute the dataset-relative logical chunk index. */
            if (H5SC__compute_logical_chunk_index(file_ndims, file_dims, chunk_dims, coords, &log_chk_idx) <
                0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTCOMPUTE, FAIL, "failed to compute linear chunk index");

            sel_chunk->chk_log_coord = log_chk_idx;

            /*
             * Construct the selection within the logical chunk.
             */
            if (H5S_SEL_ALL == file_sel_type) {
                hsize_t clip_count[H5S_MAX_RANK];
                bool    needs_clip = false;

                if (NULL == (sel_chunk->file_space = H5S_copy(single_chunk_space, true, false)))
                    HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy chunk dataspace");

                /*
                 * Compute the portion of this logical chunk that lies within the
                 * current dataset extent. This is done per chunk rather than by
                 * maintaining partial-edge state across the chunk iteration.
                 */
                for (u = 0; u < file_ndims; u++) {
                    if (coords[u] >= file_dims[u])
                        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                    "selected chunk origin lies outside dataset extent");

                    clip_count[u] = MIN(chunk_dims[u], file_dims[u] - coords[u]);

                    if (clip_count[u] == 0)
                        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "computed zero-sized chunk intersection");

                    if (clip_count[u] < chunk_dims[u])
                        needs_clip = true;
                }

                /*
                 * H5S_copy(single_chunk_space) has H5S_ALL selected. Convert partial
                 * chunks to an explicit hyperslab covering only the in-extent region.
                 */
                if (needs_clip) {
                    memset(zeros, 0, sizeof(zeros));

                    if (H5S_select_hyperslab(sel_chunk->file_space, H5S_SELECT_SET, zeros, NULL, clip_count,
                                             NULL) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL,
                                    "can't create clipped chunk selection");
                }
            }
            else {
                hsize_t clip_count[H5S_MAX_RANK];

                assert(H5S_SEL_HYPERSLABS == file_sel_type);

                for (u = 0; u < file_ndims; u++) {
                    if (coords[u] >= file_dims[u])
                        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                    "attempted to build selection for chunk outside dataset extent");

                    clip_count[u] = MIN(chunk_dims[u], file_dims[u] - coords[u]);

                    if (clip_count[u] == 0)
                        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                    "computed zero-sized clipped chunk selection");
                }

                if (H5S_combine_hyperslab(file_space, H5S_SELECT_AND, coords, NULL, clip_count, NULL,
                                          &sel_chunk->file_space) < 0)
                    HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL,
                                "unable to combine selection with clipped chunk block");

                /*
                 * Convert the selected dataspace from dataset coordinates to
                 * logical chunk coordinates.
                 */
                if (H5S_set_extent_real(sel_chunk->file_space, chunk_dims) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "can't adjust chunk dimensions");

                if (H5S_SELECT_ADJUST_U(sel_chunk->file_space, coords) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "can't adjust chunk selection");
            }

            {
                hsize_t chunk_points = H5S_GET_SELECT_NPOINTS(sel_chunk->file_space);

                if (chunk_points == 0 || chunk_points > sel_points)
                    HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                "invalid selected chunk point count during SCC selection init");

                sel_points -= chunk_points;
            }

            /*
             * Increment only after this entry is fully constructed. This
             * makes num_sel_chunks represent the set of valid entries.
             */
            sc_io_info->num_sel_chunks++;
        }

        /*
         * Advance to the next chunk in row-major chunk-coordinate order.
         */
        curr_dim = (int)file_ndims - 1;

        coords[curr_dim] += chunk_dims[curr_dim];
        end[curr_dim] += chunk_dims[curr_dim];
        scaled[curr_dim]++;

        if (coords[curr_dim] > file_sel_end[curr_dim]) {
            do {
                scaled[curr_dim] = start_scaled[curr_dim];
                coords[curr_dim] = start_coords[curr_dim];
                end[curr_dim]    = (coords[curr_dim] + chunk_dims[curr_dim]) - 1;

                curr_dim--;

                if (curr_dim >= 0) {
                    scaled[curr_dim]++;
                    coords[curr_dim] += chunk_dims[curr_dim];
                    end[curr_dim] = (coords[curr_dim] + chunk_dims[curr_dim]) - 1;
                }
            } while (curr_dim >= 0 && coords[curr_dim] > file_sel_end[curr_dim]);
        }
    }

done:
    if (single_chunk_space && H5S_close(single_chunk_space) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL,
                    "can't release temporary chunk dataspace during selection init");

    if (ret_value < 0) {
        if (sc_io_info)
            if (H5SC__io_info_reset(sc_io_info) < 0)
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't reset SCC selection information");
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__selection_io_info_init() */

/*-------------------------------------------------------------------------
 * Function: H5SC__merge_defined_chunk_selection
 *
 * Purpose:
 *   Translate a defined-value selection from logical chunk coordinates into
 *     dataset coordinates and merge it into an accumulating dataset-level
 *     defined-value selection.
 *
 *   chunk_defined describes defined elements relative to the origin of one
 *     logical chunk. The selection is translated using sel->coords and then
 *     combined with defined using H5S_SELECT_OR.
 *
 *   Selection types are handled explicitly:
 *
 *       H5S_SEL_NONE
 *         Contributes no elements and leaves the accumulator unchanged.
 *
 *       H5S_SEL_ALL
 *         Represents the complete valid portion of the logical chunk. It is
 *         converted to an explicit dataset-coordinate hyperslab and clipped
 *         against the current dataset extent before merging. This prevents
 *         an ALL selection from being reinterpreted as the entire dataset
 *         when the dataspace extent changes.
 *
 *       H5S_SEL_HYPERSLABS
 *         The chunk-local selection is copied into a dataset-sized dataspace
 *         and translated forward by the chunk's dataset-relative origin.
 *
 *   The accumulator is also handled without assuming that HDF5 preserves a
 *     particular internal selection representation. In particular, a
 *     selection that covers the complete dataspace may be canonicalized by
 *     the dataspace layer as H5S_SEL_ALL.
 *
 *   Point selections are not currently supported by the SCC interface.
 *
 * Inputs:
 *   H5S_t *defined:
 *     Dataset-sized dataspace containing the accumulated defined-value
 *     selection. Updated in place.
 *
 *   const H5S_t *chunk_defined:
 *     Dataspace containing the defined-value selection for one logical
 *     chunk, expressed in chunk-local coordinates.
 *
 *   const H5SC_io_sel_chunk_t *sel:
 *     Selected-chunk information containing the dataset-relative chunk
 *     origin in sel->coords.
 *
 *   unsigned ndims:
 *     Rank of the dataset and logical chunk.
 *
 *   const hsize_t *dset_dims:
 *     Current dataset dimensions.
 *
 *   const hsize_t *chunk_dims:
 *     Logical structured-chunk dimensions.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__merge_defined_chunk_selection(H5S_t *defined, const H5S_t *chunk_defined,
                                    const H5SC_io_sel_chunk_t *sel, unsigned ndims, const hsize_t *dset_dims,
                                    const hsize_t *chunk_dims)
{
    H5S_t       *translated = NULL;
    H5S_t       *combined   = NULL;
    H5S_sel_type sel_type;
    hssize_t     adjust[H5S_MAX_RANK];
    hsize_t      count[H5S_MAX_RANK];
    herr_t       ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(defined);
    assert(chunk_defined);
    assert(sel);
    assert(ndims > 0);
    assert(dset_dims);
    assert(chunk_dims);

    if ((sel_type = H5S_GET_SELECT_TYPE(chunk_defined)) < H5S_SEL_NONE)
        HGOTO_ERROR(H5E_DATASPACE, H5E_BADSELECT, FAIL, "unable to get defined chunk selection type");

    if (H5S_SEL_NONE == sel_type)
        HGOTO_DONE(SUCCEED);

    if (NULL == (translated = H5S_create_simple(ndims, dset_dims, NULL)))
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCREATE, FAIL,
                    "unable to create translated defined-value dataspace");

    if (H5S_select_none(translated) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, FAIL,
                    "unable to clear translated defined-value selection");

    if (H5S_SEL_ALL == sel_type) {
        /*
         * Explicitly describe this complete logical chunk as a hyperslab in
         * dataset coordinates. Clip defensively at a partial dataset edge.
         */
        for (unsigned u = 0; u < ndims; u++) {
            if (sel->coords[u] >= dset_dims[u])
                HGOTO_ERROR(H5E_DATASPACE, H5E_BADVALUE, FAIL, "defined chunk lies outside dataset extent");

            count[u] = MIN(chunk_dims[u], dset_dims[u] - sel->coords[u]);

            if (count[u] == 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_BADVALUE, FAIL,
                            "defined chunk has zero-sized dataset intersection");
        }

        if (H5S_select_hyperslab(translated, H5S_SELECT_SET, sel->coords, NULL, count, NULL) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, FAIL,
                        "unable to translate full defined chunk selection");
    }
    else if (H5S_SEL_HYPERSLABS == sel_type) {
        /*
         * Copy the logical-chunk selection into a dataset-sized dataspace.
         * H5S_SELECT_ADJUST_S subtracts the supplied adjustment, so a
         * negative chunk origin translates the selection forward into
         * dataset coordinates.
         */
        if (H5S_SELECT_COPY(translated, chunk_defined, false) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy defined chunk selection");

        for (unsigned u = 0; u < ndims; u++) {
            H5_CHECK_OVERFLOW(sel->coords[u], hsize_t, hssize_t);
            adjust[u] = -(hssize_t)sel->coords[u];
        }

        if (H5S_SELECT_ADJUST_S(translated, adjust) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSET, FAIL, "unable to translate defined chunk selection");
    }
    else
        HGOTO_ERROR(H5E_DATASPACE, H5E_UNSUPPORTED, FAIL, "unsupported defined-value selection type");

    /* Do not assume that HDF5 preserves a hyperslab representation after
     * selection copies/combinations. A selection that covers the entire
     * dataspace may be represented internally as H5S_SEL_ALL.
     */
    {
        H5S_sel_type defined_type;
        H5S_sel_type translated_type;

        if ((defined_type = H5S_GET_SELECT_TYPE(defined)) < H5S_SEL_NONE)
            HGOTO_ERROR(H5E_DATASPACE, H5E_BADSELECT, FAIL,
                        "unable to get accumulated defined selection type");

        if ((translated_type = H5S_GET_SELECT_TYPE(translated)) < H5S_SEL_NONE)
            HGOTO_ERROR(H5E_DATASPACE, H5E_BADSELECT, FAIL,
                        "unable to get translated defined selection type");

        /*
         * Nothing new to add:
         *
         *     X OR NONE == X
         */
        if (H5S_SEL_NONE == translated_type)
            HGOTO_DONE(SUCCEED);

        /*
         * The accumulated selection already covers the complete extent:
         *
         *     ALL OR X == ALL
         */
        if (H5S_SEL_ALL == defined_type)
            HGOTO_DONE(SUCCEED);

        /*
         * The new selection covers the complete extent:
         *
         *     X OR ALL == ALL
         */
        if (H5S_SEL_ALL == translated_type) {
            if (H5S_select_all(defined, true) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, FAIL,
                            "unable to set accumulated defined selection to all");

            HGOTO_DONE(SUCCEED);
        }

        /*
         * Initialize an empty accumulator directly from the translated
         * selection:
         *
         *     NONE OR X == X
         */
        if (H5S_SEL_NONE == defined_type) {
            if (H5S_SELECT_COPY(defined, translated, false) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL,
                            "unable to initialize accumulated defined selection");

            HGOTO_DONE(SUCCEED);
        }

        /*
         * Point selections are intentionally unsupported by the current SCC
         * interface. The remaining supported case should therefore consist
         * of two hyperslab selections.
         */
        if (H5S_SEL_HYPERSLABS != defined_type || H5S_SEL_HYPERSLABS != translated_type)
            HGOTO_ERROR(H5E_DATASPACE, H5E_UNSUPPORTED, FAIL,
                        "unsupported selection type while merging defined values");

        if (NULL == (combined = H5S__combine_select(defined, H5S_SELECT_OR, translated)))
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTINIT, FAIL, "unable to merge defined chunk selection");

        if (H5S_SELECT_COPY(defined, combined, false) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to update accumulated defined selection");
    }

done:
    if (combined && H5S_close(combined) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "unable to close combined defined-value dataspace");

    if (translated && H5S_close(translated) < 0)
        HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                    "unable to close translated defined-value dataspace");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__merge_defined_chunk_selection() */

/*-------------------------------------------------------------------------
 * Function: H5SC__erase_io_info_init
 *
 * Purpose:
 *   Initialize the SCC request state required by H5SC_erase().
 *
 *   File-selection decomposition is delegated to
 *     H5SC__selection_io_info_init(), which produces cache-neutral
 *     H5SC_io_sel_chunk_t entries containing chunk-local selections and
 *     logical chunk identity information.
 *
 *   This function then adds the erase-specific cache state for each selected
 *     logical chunk. It computes the SCC chunk key, finds or creates the
 *     corresponding H5SC_chunk_t entry, inserts newly created entries into
 *     the SCC chunk hash table and dataset-local LRU, synchronizes persistent
 *     scaled-coordinate information, and increments the chunk pin count.
 *
 *   On successful return, H5SC_erase() assumes responsibility for releasing
 *     the pins established here. If initialization fails after one or more
 *     chunks have been pinned, this function unwinds those pins before
 *     resetting the request selection state.
 *
 *   This function intentionally contains only the cache-specific portion of
 *     erase initialization; selection decomposition remains centralized in
 *     H5SC__selection_io_info_init() so the same logic can be reused by
 *     cache-neutral operations such as H5SC_get_defined().
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the shared chunk cache containing the dataset and chunk
 *     tracking structures.
 *
 *   H5D_t *dset:
 *     Pointer to the structured-chunk dataset being prepared for erase.
 *
 *   const H5S_t *file_space:
 *     Dataset dataspace containing the selection to erase.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__erase_io_info_init(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space)
{
    H5SC_dset_header_t *dset_hdr   = NULL;
    H5SC_io_info_t     *sc_io_info = NULL;
    herr_t              ret_value  = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(file_space);
    assert(dset->shared->layout.sc_ops);

    /*
     * Locate the dataset's SCC header. Erase requires cache-backed chunk
     * entries, unlike the common selection-decomposition helper.
     */
    dset_hdr = H5SC__ht_dset_find(cache, dset->oloc.addr);
    if (dset_hdr == NULL)
        HGOTO_ERROR(H5E_SCC, H5E_HT_NOTFOUND, FAIL,
                    "dataset should have been added to the hash table upon creation");

    dset_hdr->dset = dset;

    sc_io_info = dset_hdr->io_info;
    assert(sc_io_info);

    /*
     * Build the cache-neutral per-chunk selection information.
     */
    if (H5SC__selection_io_info_init(dset, file_space, sc_io_info) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to initialize SCC erase selection information");

    /*
     * Associate each selected logical chunk with its SCC cache entry and
     * pin that entry for the erase request.
     */
    for (size_t i = 0; i < sc_io_info->num_sel_chunks; i++) {
        H5SC_io_sel_chunk_t *sel_chunk = &sc_io_info->sel_chunks[i];
        H5SC_chunk_key_t     chk_key;
        H5SC_chunk_t        *cached_chk = NULL;

        if (H5SC__compute_chunk_key(&dset->oloc.addr, &sel_chunk->chk_log_coord, &chk_key) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CHUNK_KEY, FAIL, "failed to compute chunk key during erase init");

        cached_chk = H5SC__ht_chunk_find(cache, &chk_key);

        if (cached_chk) {
            cached_chk->chk_log_coord = sel_chunk->chk_log_coord;

            /*
             * Keep the cache entry's persistent coordinate metadata
             * synchronized with the selection entry.
             */
            cached_chk->ndims = dset->shared->ndims;
            H5MM_memcpy(cached_chk->scaled, sel_chunk->scaled, sizeof(hsize_t) * cached_chk->ndims);

            if (H5SC__chunk_pin(cache, dset_hdr, cached_chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTPIN, FAIL,
                            "unable to pin cached chunk during erase initialization");

            sel_chunk->pin_held = true;

            cached_chk->last_op = H5SC_TAG_PIN_IOINIT_CACHED;

            sel_chunk->cached_chunk = cached_chk;
        }
        else {
            cached_chk = H5SC__make_chunk(chk_key, (size_t)0, (size_t)0, false);

            if (!cached_chk)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTALLOC, FAIL, "failed to create chunk during erase init");

            /*
             * Populate persistent chunk identity/coordinate information
             * before exposing the entry through the hash table or LRU.
             */
            cached_chk->chk_log_coord = sel_chunk->chk_log_coord;
            cached_chk->ndims         = dset->shared->ndims;

            H5MM_memcpy(cached_chk->scaled, sel_chunk->scaled, sizeof(hsize_t) * cached_chk->ndims);

            if (H5SC__ht_chunk_insert(cache, cached_chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL,
                            "failed to insert chunk into hash table during erase init");

            if (H5SC__chunk_lru_prepend(cache, dset_hdr, cached_chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL,
                            "failed to insert chunk into dataset LRU during erase init");

            if (H5SC__chunk_pin(cache, dset_hdr, cached_chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTPIN, FAIL,
                            "unable to pin new chunk during erase initialization");

            sel_chunk->pin_held = true;

            cached_chk->last_op = H5SC_TAG_PIN_IOINIT;

            sel_chunk->cached_chunk = cached_chk;
        }
    }

#if H5SC_DO_SANITY_CHECKS
    if (H5SC__verify_cached_reclaimability(cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                    "cached reclaimability mismatch after erase initialization");
#endif

done:
    /*
     * H5SC_erase() owns normal pin unwinding once initialization succeeds.
     *
     * If initialization itself fails after some entries have been pinned,
     * unwind those pins here before resetting the selection state.
     */
    if (ret_value < 0 && sc_io_info) {
        for (size_t i = 0; i < sc_io_info->num_sel_chunks; i++) {
            H5SC_io_sel_chunk_t *sel = &sc_io_info->sel_chunks[i];
            H5SC_chunk_t        *chk = sel->cached_chunk;

            if (sel->pin_held) {
                if (!chk || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                    HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "unable to unwind erase initialization pin");
                else {
                    sel->pin_held = false;
                    chk->last_op  = H5SC_TAG_UNPIN_ERASE_DONE_ERROR;
                }
            }
        }

        if (H5SC__io_info_reset(sc_io_info) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't reset erase I/O information");
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__erase_io_info_init() */

/*-------------------------------------------------------------------------
 * Function: H5SC__io_info_reset
 *
 * Purpose:
 *   Reset an H5SC_io_info_t structure for reuse while retaining its
 *     sel_chunks backing allocation.
 *
 *   The function closes temporary file and memory dataspaces owned by the
 *     currently active H5SC_io_sel_chunk_t entries, clears per-request
 *     references, resets the active selection count and selection-window
 *     state, and invalidates lookup metadata across the reusable
 *     sel_chunks allocation.
 *
 *   lookup_udata requires special ownership handling. Following a successful
 *     SCC miss lookup, lookup_udata is a transient alias to layout-specific
 *     udata whose persistent ownership has been transferred to the
 *     associated H5SC_chunk_t when required. This reset routine therefore
 *     clears lookup_udata but does not free it.
 *
 *   Callers are responsible for ensuring that any lookup-generated udata
 *     has either been transferred to its owning H5SC_chunk_t or otherwise
 *     released before this function is invoked. Resident chunk udata is
 *     subsequently released through the layout client's normal chunk
 *     teardown/evict path.
 *
 *   The sel_chunks allocation itself is intentionally retained so that it
 *     may be reused by a later SCC request. H5SC__io_info_term() performs
 *     final destruction of that allocation.
 *
 *  Any H5SC_io_scratch_t allocation associated with sc_io_info is also
 *     retained unchanged for reuse by subsequent requests.
 *
 * Inputs:
 *   H5SC_io_info_t *sc_io_info:
 *     Pointer to the dataset-owned SCC request state to reset.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL if one or more owned temporary dataspaces cannot be released.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__io_info_reset(H5SC_io_info_t *sc_io_info)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(sc_io_info);

    /* Release the owned temporary dataspaces and clear active per-chunk state. */
    for (size_t i = 0; i < sc_io_info->num_sel_chunks; i++) {
        H5SC_io_sel_chunk_t *c = &sc_io_info->sel_chunks[i];

        if (c->file_space && !c->file_space_shared)
            if (H5S_close(c->file_space) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary file dataspace");

        if (c->mem_space && !c->mem_space_shared)
            if (H5S_close(c->mem_space) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary memory dataspace");

        c->file_space        = NULL;
        c->mem_space         = NULL;
        c->file_space_shared = false;
        c->mem_space_shared  = false;
        c->dset_info         = NULL;
    }

    sc_io_info->sel_order               = NULL;
    sc_io_info->num_resident_sel_chunks = 0;
    sc_io_info->sel_start               = 0;

    /* Mark the active selection set as empty. */
    sc_io_info->num_sel_chunks = 0;

    /* Clear per-invocation lookup metadata across the full allocation
     *
     * Note that lookup_udata is a non-owning, per-request alias after
     * a successful lookup. Any lookup udata that must persist beyond the
     * request is owned by the associated H5SC_chunk_t and is released
     * through the layout client's normal chunk teardown path. Callers
     * must resolve or transfer ownership of lookup-generated udata
     * before invoking this reset routine.
     * */
    if (sc_io_info->sel_chunks) {
        for (size_t i = 0; i < sc_io_info->sel_chunks_alloced; i++) {
            sc_io_info->sel_chunks[i].lookup_valid                    = false;
            sc_io_info->sel_chunks[i].lookup_addr                     = HADDR_UNDEF;
            sc_io_info->sel_chunks[i].lookup_disk_nbytes              = 0;
            sc_io_info->sel_chunks[i].lookup_defined_values_size      = 0;
            sc_io_info->sel_chunks[i].lookup_size_hint                = 0;
            sc_io_info->sel_chunks[i].lookup_defined_values_size_hint = 0;
            sc_io_info->sel_chunks[i].lookup_udata                    = NULL;
            sc_io_info->sel_chunks[i].pin_held                        = false;

            sc_io_info->sel_chunks[i].estimate_resident_size = 0;
            sc_io_info->sel_chunks[i].estimate_source        = H5SC_SIZE_EST_NONE;
            sc_io_info->sel_chunks[i].estimate_pending       = false;
        }
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__io_info_reset()*/

/*-------------------------------------------------------------------------
 * Function: H5SC__io_info_term
 *
 * Purpose:
 *   Permanently release resources owned directly by an H5SC_io_info_t
 *     structure.
 *
 *   The function closes any temporary per-chunk file or memory dataspaces
 *     owned by the currently active selection entries, frees the sel_chunks
 *     backing allocation, and resets the request bookkeeping fields.
 *
 *   The H5SC_io_info_t structure itself is not freed by this routine.
 *
 *   Layout-specific chunk objects and udata are not owned by
 *     H5SC_io_info_t and are therefore not released here. Persistent
 *     callback state belongs to the corresponding H5SC_chunk_t and must be
 *     released through the normal chunk teardown path before the dataset
 *     header is destroyed.
 *
 * Inputs:
 *   H5SC_io_info_t *sc_io_info:
 *     Pointer to the SCC request-state structure to terminate.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL if an owned temporary dataspace cannot be released.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__io_info_term(H5SC_io_info_t *sc_io_info)
{
    size_t i;
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(sc_io_info);

    /* Close any owned temporary dataspaces for active selected chunks. */
    for (i = 0; i < sc_io_info->num_sel_chunks; i++) {
        if (sc_io_info->sel_chunks[i].file_space && !sc_io_info->sel_chunks[i].file_space_shared)
            if (H5S_close(sc_io_info->sel_chunks[i].file_space) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary file dataspace");

        if (sc_io_info->sel_chunks[i].mem_space && !sc_io_info->sel_chunks[i].mem_space_shared)
            if (H5S_close(sc_io_info->sel_chunks[i].mem_space) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary memory dataspace");
    }

    if (sc_io_info->scratch) {
        H5SC__io_scratch_free_storage(sc_io_info->scratch);
        sc_io_info->scratch = (H5SC_io_scratch_t *)H5MM_xfree(sc_io_info->scratch);
    }

    /* Free backing storage and reset structure state. */
    sc_io_info->sel_chunks              = (H5SC_io_sel_chunk_t *)H5MM_xfree(sc_io_info->sel_chunks);
    sc_io_info->scratch                 = (H5SC_io_scratch_t *)H5MM_xfree(sc_io_info->scratch);
    sc_io_info->sel_order               = NULL;
    sc_io_info->sel_start               = 0;
    sc_io_info->num_sel_chunks          = 0;
    sc_io_info->num_resident_sel_chunks = 0;
    sc_io_info->sel_chunks_alloced      = 0;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__io_info_term() */

/*-------------------------------------------------------------------------
 * Function: H5SC_read
 *
 * Purpose:
 *   Function that performs a single dataset read operation through the
 *     SCC using H5SC_io_info_t state prepared earlier by
 *     H5SC__io_info_init(). The function processes the active
 *     selected-chunk window for the dataset, materializes any needed
 *     resident chunk state from cache or disk, scatters chunk data into
 *     the user buffer, updates SCC chunk state, and releases the
 *     per-selection pins established during I/O setup.
 *
 *   For each selected chunk in the active window, the function uses cached
 *     lookup metadata to distinguish resident cache hits from misses,
 *     creates fill-value-backed shell chunks for chunks not present on
 *     disk, reads and decodes on-disk chunks when necessary, and then
 *     invokes the layout client's scatter_mem callback to transfer data
 *     from the chunk representation into the caller's memory selection.
 *
 *   The function follows the current SCC selection-window semantics:
 *     sel_chunks remains the stable base allocation owned by the dataset,
 *     while sel_start and num_sel_chunks define the active subrange to
 *     process for the current call.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state used for lookup, residency,
 *     accounting, and callback-driven chunk processing.
 *
 *   H5D_dset_io_info_t *dset_info:
 *     Pointer to the array of dataset I/O request descriptors to process.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

herr_t
H5SC_read(H5SC_t *cache, H5D_dset_io_info_t *dset_info)
{

    herr_t             ret_value = SUCCEED;
    H5D_io_type_info_t my_io_type_info;    /* First used in the scatter_mem callback */
    const H5S_t       *scatter_mem_space;  /* Used in the scatter_mem callback */
    const H5S_t       *scatter_file_space; /* Used in the scatter_mem callback */
    haddr_t            md_tag                                  = HADDR_UNDEF;
    bool               md_tag_set                              = false;
    bool               partial_bound_chunks_different_encoding = false;
    H5O_stc_pline_t   *pline                                   = NULL; /* I/O pipeline info */
    hbool_t            filtered                                = false;
    size_t             alloc_size_total;

    H5SC_dset_header_t  *dset_hdr                 = NULL;
    H5SC_io_info_t      *sc_io_info               = NULL;
    H5SC_io_sel_chunk_t *base                     = NULL;
    size_t               sel_start                = 0;
    size_t               sel_count                = 0;
    size_t               chunk_count              = 0;
    size_t               old_dset_size            = 0;
    const hsize_t      **scaled                   = NULL;
    haddr_t            **addr                     = NULL;
    hsize_t            **size                     = NULL;
    hsize_t            **defined_values_size      = NULL;
    size_t             **size_hint                = NULL;
    size_t             **defined_values_size_hint = NULL;
    void               **udata_arr                = NULL;
    size_t              *miss_arr                 = NULL;
    void               **chunk_arr                = NULL;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    dset_hdr = H5SC__ht_dset_find(cache, dset_info[0].dset->oloc.addr);
    if (dset_hdr == NULL) {
        HGOTO_ERROR(H5E_SCC, H5E_HT_NOTFOUND, FAIL,
                    "dataset should have been added to the hash table upon creation.");
    }

    /* Use the dataset-owned io_info */
    sc_io_info = dset_hdr->io_info;
    assert(sc_io_info);

    base      = sc_io_info->sel_chunks;
    sel_start = sc_io_info->sel_start;
    sel_count = sc_io_info->num_sel_chunks;

    /* Throw an error if no chunks were selected */
    if (sc_io_info->num_sel_chunks == 0) {
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGETSIZE, FAIL,
                    "The number of selected structured chunks should be non-zero (SCC)");
    }

    if (sel_start > sc_io_info->sel_chunks_alloced ||
        sc_io_info->num_sel_chunks > (sc_io_info->sel_chunks_alloced - sel_start)) {
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "invalid selection window (sel_start/num_sel_chunks)");
    }

    assert(sel_count > 0);
    assert(sel_start + sel_count <= sc_io_info->sel_chunks_alloced);

    /* Sanity checks to ensure the SCC callbacks are defined for this dataset. */
    assert(dset_info[0].dset->shared->layout.sc_ops->lookup);
    assert(dset_info[0].dset->shared->layout.sc_ops->decode);
    assert(dset_info[0].dset->shared->layout.sc_ops->scatter_mem);

    /* Snapshot dataset accounting once per request; update after I/O completes */
    old_dset_size = dset_hdr->curr_dset_size;

    /* Set metadata tagging for this dataset */
    H5AC_tag(dset_info[0].dset->oloc.addr, &md_tag);
    md_tag_set = true;

    chunk_count = sc_io_info->num_sel_chunks;

    /*
     * Ensure reusable callback-vector scratch is large enough for this active
     * selection window.
     */
    if (H5SC__io_scratch_ensure(sc_io_info, sc_io_info->num_sel_chunks) < 0)
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to prepare reusable SCC read scratch storage");

    assert(sc_io_info->scratch);
    assert(sc_io_info->scratch->alloced >= chunk_count);

    addr                     = sc_io_info->scratch->addr;
    size                     = sc_io_info->scratch->size;
    defined_values_size      = sc_io_info->scratch->defined_values_size;
    size_hint                = sc_io_info->scratch->size_hint;
    defined_values_size_hint = sc_io_info->scratch->defined_values_size_hint;
    scaled                   = sc_io_info->scratch->scaled;
    udata_arr                = sc_io_info->scratch->udata;
    miss_arr                 = sc_io_info->scratch->miss_idx;
    chunk_arr                = sc_io_info->scratch->chunk;

    /* Chunk lookup (in file) */

    /* For each chunk in this dataset, initialize each necessary array value*/
    for (size_t j = 0; j < chunk_count; j++) {
        /* Setup scaled for the jth chunk. */
        scaled[j] = H5SC__io_sel_at(sc_io_info, j)->scaled;

        /*
         * The pointer views were permanently bound to reusable contiguous
         * backing storage by H5SC__io_scratch_ensure(). Reset only the scalar
         * values needed by this call.
         */
        *addr[j]                     = HADDR_UNDEF;
        *size[j]                     = 0;
        *defined_values_size[j]      = 0;
        *size_hint[j]                = 0;
        *defined_values_size_hint[j] = 0;

        udata_arr[j] = NULL;
        chunk_arr[j] = NULL;
    }

    /* Partition chunks into cache-hits (resident decoded chunks) and misses (need to be looked up on
     * disk) */
    size_t miss_count = 0;

    /* Populate lookup-related information for cache misses */
    if (H5SC__lookup_cache_misses(dset_info[0].dset, sc_io_info) < 0) {
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to lookup cache missed chunks (SCC)");
    }

    for (size_t j = 0; j < chunk_count; j++) {

        H5SC_chunk_t *chk = H5SC__io_sel_at(sc_io_info, j)->cached_chunk;

        if (chk && chk->chunk_obj) {
            /* Cache hit: Skip on-disk lookup */
            *addr[j]      = chk->disk_addr;
            *size[j]      = (size_t)chk->disk_nbytes;
            *size_hint[j] = (size_t)chk->disk_nbytes;

            *defined_values_size[j]      = 0;
            *defined_values_size_hint[j] = 0;

            udata_arr[j] = chk->udata;
        }
        else {
            miss_arr[miss_count++] = j;
        }
    }

    /* Fill lookup outputs from the cache missed chunks */
    for (size_t j = 0; j < miss_count; j++) {
        size_t idx = miss_arr[j];

        H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, idx);
        assert(sel->lookup_valid == true);

        *addr[idx]                     = sel->lookup_addr;
        *size[idx]                     = sel->lookup_disk_nbytes;
        *defined_values_size[idx]      = sel->lookup_defined_values_size;
        *size_hint[idx]                = sel->lookup_size_hint;
        *defined_values_size_hint[idx] = sel->lookup_defined_values_size_hint;
        udata_arr[idx]                 = sel->lookup_udata;

        if (!H5_addr_defined(*addr[idx]) && *size_hint[idx] == 0) {
            *size_hint[idx] = (size_t)dset_info[0].dset->shared->layout.u.struct_chunk.size;
        }
    }

    /* partial_bound_chunks_different_encoding:
     * When enabled, filters are not applied to partial edge chunks. When disabled, partial edge chunks
     * are filtered. Enabling this option will improve performance when appending to the dataset and, when
     * compression filters are used, prevent reallocation of these chunks.
     */
    if (dset_info[0].dset->shared->layout.sc_ops->layout_query(dset_info[0].dset, NULL, NULL,
                                                               &partial_bound_chunks_different_encoding) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query chunk dimensions");

    /* Filtered or not */
    pline = &(dset_info[0].dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects)
        filtered = true;

    /* For each chunk in this dataset: */
    for (size_t j = 0; j < chunk_count; j++) {
        /* true: a NOT-to-be-filtered-partial-edge chunk */
        /* false : a to-be-filtered-partial-edge-chunk */
        bool                 partial_bound = false;
        H5SC_io_sel_chunk_t *sel_chunk;
        H5SC_chunk_t        *cached_chunk;
        size_t               nbytes_local = 0;

        my_io_type_info.tconv_buf      = NULL; /* Pointer to the datatype conv buffer */
        my_io_type_info.tconv_buf_size = dset_info[0].type_info.src_type_size;
        my_io_type_info.bkg_buf        = NULL; /* Pointer to background buffer */
        my_io_type_info.bkg_buf_size   = dset_info[0].type_info.dst_type_size;

        sel_chunk    = H5SC__io_sel_at(sc_io_info, j);
        cached_chunk = sel_chunk->cached_chunk;
        assert(cached_chunk);

        scatter_mem_space  = sel_chunk->mem_space;
        scatter_file_space = sel_chunk->file_space;

        /* Cache hit: use the resident decoded chunk object */
        if (cached_chunk->chunk_obj) {
            chunk_arr[j] = cached_chunk->chunk_obj;
            if (cached_chunk->udata) {
                if (udata_arr[j] && udata_arr[j] != cached_chunk->udata) {
                    udata_arr[j] = H5MM_xfree(udata_arr[j]);
                }
                udata_arr[j] = cached_chunk->udata;
            }
        }

        /* Free the invalid udata created by the lookup callback. */
        if (!cached_chunk->chunk_obj && !H5_addr_defined(*addr[j])) {

            if (!size_hint[j] || *size_hint[j] == 0) {
                HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                            "read: invalid size_hint for new/resident chunk allocation");
            }

            /* If the chunk is not present within the cache, it must be allocated/created here */
            if (!(H5SC__io_sel_at(sc_io_info, j)->cached_chunk &&
                  H5SC__io_sel_at(sc_io_info, j)->cached_chunk->udata == udata_arr[j])) {
                udata_arr[j] = H5MM_xfree(udata_arr[j]);
            }

            /* Create a new chunk that will then have the fill value written to it. */
            if (dset_info[0].dset->shared->layout.sc_ops->new_chunk(dset_info[0].dset, false, &nbytes_local,
                                                                    size_hint[j], &chunk_arr[j],
                                                                    &udata_arr[j]) < 0) {
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to create new chunk (SCC)");
            }

            alloc_size_total = 0;

            if (dset_info[0].dset->shared->layout.sc_ops->fill(
                    &dset_info[0], &my_io_type_info, H5SC__io_sel_at(sc_io_info, j)->file_space,
                    &nbytes_local, size_hint[j], &alloc_size_total, chunk_arr[j], udata_arr[j]) < 0) {
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to fill chunk (SCC)");
            }
            /* Add info for the resident chunk within the cache */
            cached_chunk->chunk_obj   = chunk_arr[j];
            cached_chunk->udata       = udata_arr[j];
            cached_chunk->disk_addr   = HADDR_UNDEF;
            cached_chunk->disk_nbytes = 0;

            if (H5SC__chunk_set_dirty(cache, dset_hdr, cached_chunk, false) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "unable to set materialized read chunk clean");

            /* Update per-dataset bytes now; global accounting is reconciled once per request */
            if (H5SC__chunk_update_cached_size(cache, dset_hdr, cached_chunk, alloc_size_total) < 0) {
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "read: chunk size accounting failed");
            }

            if (sel_chunk->estimate_pending) {
#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
                H5SC__stats_record_size_estimate(cache, sel_chunk->estimate_source,
                                                 sel_chunk->estimate_resident_size, alloc_size_total);
#endif

                sel_chunk->estimate_pending       = false;
                sel_chunk->estimate_source        = H5SC_SIZE_EST_NONE;
                sel_chunk->estimate_resident_size = 0;
            }

            H5SC__update_resident_estimate_history(dset_hdr, alloc_size_total);
        }

        /* If the chunk lookup is successful; only read/decode if the chunk is not already a resident
         * in the cache: */
        if (!cached_chunk->chunk_obj && H5_addr_defined(*addr[j])) {

            if (!size_hint[j] || *size_hint[j] == 0) {
                HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                            "read: invalid size_hint for decoded chunk allocation");
            }

            if (filtered && partial_bound_chunks_different_encoding &&
                H5D__chunk_is_partial_edge_chunk(dset_info[0].dset->shared->ndims,
                                                 dset_info[0].dset->shared->layout.u.struct_chunk.dim,
                                                 scaled[j], dset_info[0].dset->shared->curr_dims)) {
                partial_bound = true;
            }

            /* Allocate buffer for the chunk data */
            if (NULL == (chunk_arr[j] = H5MM_malloc(*size_hint[j]))) {
                HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, H5_ITER_ERROR,
                            "memory allocation failed for raw data chunk (SCC)");
            }

            if (H5F_block_read(dset_info[0].dset->oloc.file, H5FD_MEM_DRAW, *addr[j], *size[j],
                               chunk_arr[j]) < 0) {
                HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "unable to block read from file (SCC)");
            }

            H5_CHECKED_ASSIGN(nbytes_local, size_t, *size[j], hsize_t);

            /* Skip for invalid addr */
            if (dset_info[0].dset->shared->layout.sc_ops->decode(dset_info[0].dset, &nbytes_local,
                                                                 size_hint[j], partial_bound, &chunk_arr[j],
                                                                 udata_arr[j]) < 0) {
            }

            /* Add info for the resident chunk within the cache */
            cached_chunk->chunk_obj = chunk_arr[j];
            cached_chunk->udata     = udata_arr[j];
            cached_chunk->disk_addr = *addr[j];

            if (H5SC__chunk_set_dirty(cache, dset_hdr, cached_chunk, false) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "unable to set materialized read chunk clean");

            /* Update LRU byte counters + cache quiescent bytes */
            if (H5SC__chunk_update_cached_size(cache, dset_hdr, cached_chunk, *size_hint[j]) < 0) {
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                            "read: chunk size accounting failed after decode");
            }

            if (sel_chunk->estimate_pending) {
#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
                H5SC__stats_record_size_estimate(cache, sel_chunk->estimate_source,
                                                 sel_chunk->estimate_resident_size, *size_hint[j]);
#endif

                sel_chunk->estimate_pending       = false;
                sel_chunk->estimate_source        = H5SC_SIZE_EST_NONE;
                sel_chunk->estimate_resident_size = 0;
            }

            H5SC__update_resident_estimate_history(dset_hdr, *size_hint[j]);
        }

        if (dset_info[0].dset->shared->layout.sc_ops->scatter_mem(&dset_info[0], &my_io_type_info,
                                                                  scatter_mem_space, scatter_file_space,
                                                                  chunk_arr[j], udata_arr[j]) < 0) {
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to scatter mem for read chunk (SCC)");
        }

        {
            H5SC_chunk_t *chk = H5SC__io_sel_at(sc_io_info, j)->cached_chunk;
            assert(chk);

            if (H5SC__io_sel_at(sc_io_info, j)->pin_held) {
                if (H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                    HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "unable to release SCC request pin");

                H5SC__io_sel_at(sc_io_info, j)->pin_held = false;
                chk->last_op                             = H5SC_TAG_UNPIN_READ_DONE;
            }
            else {
                HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                            "SCC read: selected chunk does not own a valid pin");
            }
        }

    } /* Chunk Processing Loop End */

    /* Reconcile dataset/global cache size accounting once per request */
    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "global size accounting failed (SCC read)");
    }

#if H5SC_DO_SANITY_CHECKS
    if (H5SC__verify_cached_reclaimability(cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "cached reclaimability mismatch after SCC read");
#endif

done:
    if (md_tag_set)
        H5AC_tag(md_tag, NULL); /* Reset the metadata tag for the next dataset */

    if (ret_value < 0) {
        if (sc_io_info && base && sel_count > 0) {
            for (size_t j = 0; j < sel_count; j++) {
                H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, j);
                H5SC_chunk_t        *chk = sel->cached_chunk;

                if (sel->pin_held) {
                    if (!chk || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                        HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                    "SCC read cleanup encountered an invalid owned pin");
                    else {
                        sel->pin_held = false;
                        chk->last_op  = H5SC_TAG_UNPIN_READ_DONE_ERROR;
                    }
                }
            }
        }
    }

    /* Restore base sel_chunks pointer for the caller (invoke batching expects this) */
    if (sc_io_info && base)
        sc_io_info->sel_chunks = base;

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_read() */

/*-------------------------------------------------------------------------
 * Function: H5SC_write
 *
 * Purpose:
 *   Function that performs a single dataset write operation through the
 *     SCC using H5SC_io_info_t state prepared earlier by
 *     H5SC__io_info_init(). The function processes the active
 *     selected-chunk window for the dataset, materializes any needed
 *     resident chunk state from cache or disk, gathers data from the
 *     caller's memory selection into chunk buffers, marks the resident
 *     chunks dirty, updates SCC accounting, and releases the
 *     per-selection pins established during I/O setup.
 *
 *   For each selected chunk in the active window, the function uses
 *     cached lookup metadata to distinguish resident cache hits from
 *     misses, creates new resident chunks for chunks not present on disk,
 *     reads and decodes on-disk chunks when necessary, invokes the layout
 *     client's gather_mem callback to update chunk contents from the user
 *     buffer, and retains the modified resident representation as the
 *     authoritative state until an explicit flush or dirty eviction writes
 *     it to disk.
 *
 *   The function follows the current SCC selection-window semantics:
 *     sel_chunks remains the stable base allocation owned by the dataset,
 *     while sel_start and num_sel_chunks define the active subrange to
 *     process for the current call.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state used for lookup, residency,
 *     accounting, and callback-driven chunk processing.
 *
 *   H5D_dset_io_info_t *dset_info:
 *     Pointer to the dataset I/O request descriptor to process.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

herr_t
H5SC_write(H5SC_t *cache, H5D_dset_io_info_t *dset_info)
{

    herr_t               ret_value = SUCCEED;
    size_t               alloc_size_total;
    H5D_io_type_info_t   my_io_type_info; /* Used in gather_mem callback */
    const H5S_t         *gather_mem_space;
    const H5S_t         *gather_file_space;
    haddr_t              md_tag                                  = HADDR_UNDEF;
    bool                 md_tag_set                              = false;
    bool                 partial_bound_chunks_different_encoding = false;
    H5O_stc_pline_t     *pline                                   = NULL; /* I/O pipeline info */
    hbool_t              filtered                                = false;
    H5SC_dset_header_t  *dset_hdr                                = NULL;
    H5SC_io_info_t      *sc_io_info                              = NULL;
    H5SC_io_sel_chunk_t *base                                    = NULL;
    size_t               sel_start                               = 0;
    size_t               sel_count                               = 0;
    size_t               chunk_count                             = 0;
    size_t               old_dset_size                           = 0;
    const hsize_t      **scaled                                  = NULL;
    haddr_t            **addr                                    = NULL;
    hsize_t            **size                                    = NULL;
    hsize_t            **defined_values_size                     = NULL;
    size_t             **size_hint                               = NULL;
    size_t             **defined_values_size_hint                = NULL;
    void               **udata_arr                               = NULL;
    size_t              *miss_arr                                = NULL;
    void               **chunk_arr                               = NULL;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    dset_hdr = H5SC__ht_dset_find(cache, dset_info[0].dset->oloc.addr);
    if (dset_hdr == NULL) {
        HGOTO_ERROR(H5E_SCC, H5E_HT_NOTFOUND, FAIL,
                    "dataset should have been added to the hash table upon creation.");
    }

    /* Use the dataset-owned io_info */
    sc_io_info = dset_hdr->io_info;
    assert(sc_io_info);

    base      = sc_io_info->sel_chunks;
    sel_start = sc_io_info->sel_start;
    sel_count = sc_io_info->num_sel_chunks;

    assert(base);

    if (sc_io_info->num_sel_chunks == 0) {
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGETSIZE, FAIL,
                    "The number of selected structured chunks should be non-zero (SCC)");
    }

    if (sel_start > sc_io_info->sel_chunks_alloced ||
        sel_count > (sc_io_info->sel_chunks_alloced - sel_start)) {
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "invalid selection window (sel_start/num_sel_chunks)");
    }

    assert(sel_count > 0);
    assert(sel_start + sel_count <= sc_io_info->sel_chunks_alloced);

    /* Sanity checks for the SCC callbacks for this dataset. */
    assert(dset_info[0].dset->shared->layout.sc_ops->lookup);
    assert(dset_info[0].dset->shared->layout.sc_ops->new_chunk);
    assert(dset_info[0].dset->shared->layout.sc_ops->gather_mem);

    /* Snapshot dataset accounting once per request; update after I/O completes */
    old_dset_size = dset_hdr->curr_dset_size;

    /* Initialize type-conversion buffers for gather_mem(). */
    my_io_type_info.tconv_buf              = NULL; /* Datatype conv buffer (pointer) */
    my_io_type_info.tconv_buf_size         = dset_info[0].type_info.src_type_size;
    my_io_type_info.bkg_buf                = NULL; /* Pointer to background buffer */
    my_io_type_info.bkg_buf_size           = dset_info[0].type_info.dst_type_size;
    my_io_type_info.may_use_in_place_tconv = true; /* Use in-place if possible */

    /* Tag metadata accesses for this dataset. */
    H5AC_tag(dset_info[0].dset->oloc.addr, &md_tag); /* Set the metadata tag */
    md_tag_set = true;

    /* Operate only on the chunks in the active window */
    chunk_count = sel_count;

    /*
     * Ensure reusable callback-vector scratch is large enough for this active
     * write selection window.
     */
    if (H5SC__io_scratch_ensure(sc_io_info, chunk_count) < 0)
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to prepare reusable SCC write scratch storage");

    assert(sc_io_info->scratch);
    assert(sc_io_info->scratch->alloced >= chunk_count);

    scaled                   = sc_io_info->scratch->scaled;
    addr                     = sc_io_info->scratch->addr;
    size                     = sc_io_info->scratch->size;
    defined_values_size      = sc_io_info->scratch->defined_values_size;
    size_hint                = sc_io_info->scratch->size_hint;
    defined_values_size_hint = sc_io_info->scratch->defined_values_size_hint;

    udata_arr = sc_io_info->scratch->udata;
    miss_arr  = sc_io_info->scratch->miss_idx;
    chunk_arr = sc_io_info->scratch->chunk;

    /* Initialize per-chunk scratch state. */
    for (size_t j = 0; j < chunk_count; j++) {
        scaled[j] = H5SC__io_sel_at(sc_io_info, j)->scaled;

        /*
         * Callback-facing pointer views are permanently bound to reusable
         * contiguous backing storage by H5SC__io_scratch_ensure().
         * Reinitialize only the scalar values used by this write.
         */
        *addr[j]                     = HADDR_UNDEF;
        *size[j]                     = 0;
        *defined_values_size[j]      = 0;
        *size_hint[j]                = 0;
        *defined_values_size_hint[j] = 0;

        udata_arr[j] = NULL;
        chunk_arr[j] = NULL;
    }

    /* Populate on-disk metadata only for nonresident chunks. */
    size_t miss_count = 0;

    if (H5SC__lookup_cache_misses(dset_info[0].dset, sc_io_info) < 0) {
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to lookup cache missed chunks (SCC)");
    }

    for (size_t j = 0; j < chunk_count; j++) {
        H5SC_chunk_t *chk = H5SC__io_sel_at(sc_io_info, j)->cached_chunk;

        if (chk && chk->chunk_obj) {
            /* Cache hit: use the resident object and its persistent udata. */
            *addr[j]                     = chk->disk_addr;
            *size[j]                     = (hsize_t)chk->disk_nbytes;
            *size_hint[j]                = chk->cached_chunk_size;
            *defined_values_size[j]      = 0;
            *defined_values_size_hint[j] = 0;
            udata_arr[j]                 = chk->udata;
        }
        else {
            miss_arr[miss_count++] = j;
        }
    }

    /* Transfer cached lookup outputs into the general callback vectors. */
    for (size_t j = 0; j < miss_count; j++) {
        size_t               idx = miss_arr[j];
        H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, idx);

        assert(sel->lookup_valid == true);

        *addr[idx]                     = sel->lookup_addr;
        *size[idx]                     = sel->lookup_disk_nbytes;
        *defined_values_size[idx]      = sel->lookup_defined_values_size;
        *size_hint[idx]                = sel->lookup_size_hint;
        *defined_values_size_hint[idx] = sel->lookup_defined_values_size_hint;
        udata_arr[idx]                 = sel->lookup_udata;

        if (!H5_addr_defined(*addr[idx]) && *size_hint[idx] == 0)
            *size_hint[idx] = (size_t)dset_info[0].dset->shared->layout.u.struct_chunk.size;
    }

    /* Determine whether partial edge chunks use alternate encoding behavior. */
    if (dset_info[0].dset->shared->layout.sc_ops->layout_query(dset_info[0].dset, NULL, NULL,
                                                               &partial_bound_chunks_different_encoding) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query chunk dimensions");

    pline = &(dset_info[0].dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects)
        filtered = true;

    for (size_t j = 0; j < chunk_count; j++) {
        bool          partial_bound = false;
        H5SC_chunk_t *chk           = H5SC__io_sel_at(sc_io_info, j)->cached_chunk;
        size_t        nbytes_local  = 0;

        assert(chk);

        /* Persist minimal chunk identity for later flush */
        chk->ndims = dset_info[0].dset->shared->ndims;
        H5MM_memcpy(chk->scaled, scaled[j], sizeof(hsize_t) * chk->ndims);

        if (!H5_addr_defined(*addr[j])) {
            /*--------------------------------------------------------------*/
            /* No on-disk chunk exists                                      */
            /*--------------------------------------------------------------*/

            if (chk->chunk_obj) {
                /*
                 * Resident shell chunk: keep using the existing decoded chunk
                 * object. No allocation via new_chunk() is needed.
                 */
                chunk_arr[j]     = chk->chunk_obj;
                chk->disk_addr   = HADDR_UNDEF;
                chk->disk_nbytes = 0;
            }
            else {
                /*
                 * True sparse miss: no on-disk chunk and no resident chunk.
                 * Create a new resident chunk.
                 */
                if (!(H5SC__io_sel_at(sc_io_info, j)->cached_chunk &&
                      H5SC__io_sel_at(sc_io_info, j)->cached_chunk->udata == udata_arr[j])) {
                    udata_arr[j] = H5MM_xfree(udata_arr[j]);
                }

                if (!size_hint[j] || *size_hint[j] == 0) {
                    HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                "write: invalid size_hint for new chunk allocation");
                }

                if (dset_info[0].dset->shared->layout.sc_ops->new_chunk(dset_info[0].dset, false,
                                                                        &nbytes_local, size_hint[j],
                                                                        &chunk_arr[j], &udata_arr[j]) < 0) {
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to create new chunk (SCC)");
                }

                H5_CHECKED_ASSIGN(*size[j], hsize_t, nbytes_local, size_t);

                chk->chunk_obj   = chunk_arr[j];
                chk->udata       = udata_arr[j];
                chk->disk_addr   = HADDR_UNDEF;
                chk->disk_nbytes = 0;

                if (H5SC__chunk_set_dirty(cache, dset_hdr, chk, true) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "unable to mark new resident chunk dirty");

                {
                    size_t resident_bytes = *size_hint[j];
                    if (resident_bytes == 0)
                        resident_bytes = (size_t)dset_info[0].dset->shared->layout.u.struct_chunk.size;

                    if (H5SC__chunk_update_cached_size(cache, dset_hdr, chk, resident_bytes) < 0)
                        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                                    "write: chunk size accounting failed for new resident chunk");
                }
            }
        }
        else {
            chk->disk_addr = *addr[j];
            H5_CHECKED_ASSIGN(chk->disk_nbytes, size_t, *size[j], hsize_t);

            if (filtered && partial_bound_chunks_different_encoding &&
                H5D__chunk_is_partial_edge_chunk(dset_info[0].dset->shared->ndims,
                                                 dset_info[0].dset->shared->layout.u.struct_chunk.dim,
                                                 scaled[j], dset_info[0].dset->shared->curr_dims))
                partial_bound = true;

            /* Read and decode the on-disk chunk into resident form. */
            if (chk->chunk_obj) {
                /* Cache hit: decoded chunk is already resident, skip read+decode */
                chunk_arr[j] = chk->chunk_obj;
            }
            else {
                /* Cache miss: read encoded buffer, then decode into a decoded chunk object */

                if (!size_hint[j] || *size_hint[j] == 0)
                    HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                "write: invalid size_hint for decoded chunk allocation");

                assert(size_hint[j]);

                if (NULL == (chunk_arr[j] = H5MM_malloc(*size_hint[j]))) {
                    HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, H5_ITER_ERROR,
                                "memory allocation failed for raw data chunk (SCC)");
                }

                if (H5F_block_read(dset_info[0].dset->oloc.file, H5FD_MEM_DRAW, *addr[j], *size[j],
                                   chunk_arr[j]) < 0) {
                    HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "unable to block read from file (SCC)");
                }

                H5_CHECKED_ASSIGN(nbytes_local, size_t, *size[j], hsize_t);

                if (dset_info[0].dset->shared->layout.sc_ops->decode(dset_info[0].dset, &nbytes_local,
                                                                     size_hint[j], partial_bound,
                                                                     &chunk_arr[j], udata_arr[j]) < 0) {
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to decode chunk in place (SCC)");
                }

                H5_CHECKED_ASSIGN(*size[j], hsize_t, nbytes_local, size_t);

                /* After decode, chunk_arr[j] is in the in-cache format */
                chk->chunk_obj = chunk_arr[j];
                chk->udata     = udata_arr[j];

                if (H5SC__chunk_set_dirty(cache, dset_hdr, chk, false) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                                "unable to set materialized read chunk clean");

                /* Charge resident bytes for this chunk */
                size_t resident_bytes = *size_hint[j];
                if (resident_bytes == 0) {
                    resident_bytes = (size_t)dset_info[0].dset->shared->layout.u.struct_chunk.size;
                }

                if (H5SC__chunk_update_cached_size(cache, dset_hdr, chk, resident_bytes) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                                "write: chunk size accounting failed for decoded resident chunk");
            }
        }
    }

    for (size_t j = 0; j < chunk_count; j++) {
        H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, j);
        H5SC_chunk_t        *chk = sel->cached_chunk;
        size_t               nbytes_local;

        assert(sel);
        assert(chk);

        /*
         * The gather callback may begin modifying the decoded object before
         * reporting an error. Mark the chunk dirty before entering the callback so
         * that a partially modified resident object can never be treated as clean.
         */
        if (H5SC__chunk_set_dirty(cache, dset_hdr, chk, true) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "unable to mark new write chunk dirty");

        /* Gather user data into the resident chunk buffer. */
        gather_mem_space  = sel->mem_space;
        gather_file_space = sel->file_space;

#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
        {
            size_t pre_gather_size = chk->cached_chunk_size;
#endif

            alloc_size_total = 0;

            alloc_size_total = 0;

            H5_CHECKED_ASSIGN(nbytes_local, size_t, *size[j], hsize_t);

            if (dset_info[0].dset->shared->layout.sc_ops->gather_mem(
                    &dset_info[0], &my_io_type_info, gather_mem_space, gather_file_space, &nbytes_local,
                    size_hint[j], &alloc_size_total, chunk_arr[j], udata_arr[j]) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL,
                            "unable to gather memory data into resident chunk (SCC)");

            H5_CHECKED_ASSIGN(*size[j], hsize_t, nbytes_local, size_t);

#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
            if (alloc_size_total > pre_gather_size) {
                size_t growth = alloc_size_total - pre_gather_size;

                cache->stats.write_growth_count++;
                H5SC__saturating_add_size(&cache->stats.write_growth_total, growth);

                if (growth > cache->stats.write_growth_max)
                    cache->stats.write_growth_max = growth;
            }
        }
#endif

        /* gather_mem() may resize the resident representation. */
        if (H5SC__chunk_update_cached_size(cache, dset_hdr, chk, alloc_size_total) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                        "write: chunk size accounting failed after gather_mem");

        if (sel->estimate_pending) {
#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)

            H5SC__stats_record_size_estimate(cache, sel->estimate_source, sel->estimate_resident_size,
                                             alloc_size_total);
#endif

            sel->estimate_pending       = false;
            sel->estimate_source        = H5SC_SIZE_EST_NONE;
            sel->estimate_resident_size = 0;
        }

        H5SC__update_resident_estimate_history(dset_hdr, alloc_size_total);

        /*
         * Release exactly the pin owned by this selected-chunk entry.
         */
        if (sel->pin_held) {
            if (H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "unable to release SCC request pin");

            sel->pin_held = false;
            chk->last_op  = H5SC_TAG_UNPIN_WRITE_DONE;
        }
        else
            HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "SCC write: selected chunk does not own a valid pin");
    }
    /* Reconcile dataset/global cache size accounting once per request */
    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "global size accounting failed (SCC write)");
    }

    /* Close any temporary per-call dataspaces for the active window. */
    for (size_t j = 0; j < chunk_count; j++) {
        H5SC_io_sel_chunk_t *chk = H5SC__io_sel_at(sc_io_info, j);

        if (chk->file_space && !chk->file_space_shared)
            if (H5S_close(chk->file_space) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL,
                            "can't release temporary file dataspace (SCC write)");

        if (chk->mem_space && !chk->mem_space_shared)
            if (H5S_close(chk->mem_space) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL,
                            "can't release temporary memory dataspace (SCC write)");

        /* Poison only the ephemeral dataspace pointers; leave lookup cache intact */
        chk->file_space        = NULL;
        chk->mem_space         = NULL;
        chk->file_space_shared = false;
        chk->mem_space_shared  = false;
        chk->dset_info         = NULL;
    }

#if H5SC_DO_SANITY_CHECKS
    if (H5SC__verify_cached_reclaimability(cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "cached reclaimability mismatch after SCC write");
#endif

done:
    if (md_tag_set)
        H5AC_tag(md_tag, NULL); /* Reset the metadata tag for the next dataset */

    /*
     * Defensive unpin on error paths: if we abort mid-write, make sure any
     * chunks referenced by the current selection window are no longer marked
     * in-flight.
     */
    if (ret_value < 0) {
        if (sc_io_info && sc_io_info->sel_chunks && sel_count > 0) {
            for (size_t j = 0; j < sel_count; j++) {
                H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, j);
                H5SC_chunk_t        *chk = sel->cached_chunk;

                if (sel->pin_held) {
                    if (!chk || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                        HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                    "SCC write cleanup encountered an invalid owned pin");
                    else {
                        sel->pin_held = false;
                        chk->last_op  = H5SC_TAG_UNPIN_WRITE_DONE_ERROR;
                    }
                }
            }
        }
    }
    /* Restore the stable selection base pointer. */
    if (sc_io_info) {
        /*
         * base/saved_sel_count are function-scope now, so this is always valid.
         * If we error before setting them in the loop, base will be NULL.
         */
        if (base)
            sc_io_info->sel_chunks = base;
        /* num_sel_chunks and sel_start are owned by the caller’s batching logic */
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_write() */

/*-------------------------------------------------------------------------
 * Function: H5SC_direct_chunk_read
 *
 * Purpose:  Reads the chunk that starts at coordinates give
 * by offset directly from disk to buf, without any decoding
 * or conversion. First flushes that chunk if it is dirty in
 * the cache.
 *
 * Not currently implemented or supported.
 *
 * Return:   SUCCEED on success, FAIL on failure
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_direct_chunk_read(H5SC_t *cache, H5D_t *dset, const hsize_t *offset, H5_ATTR_UNUSED void *udata,
                       void *buf, size_t *buf_size)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset->shared->layout.sc_ops);
    assert(offset);
    assert(buf);
    assert(buf_size);

    HGOTO_ERROR(H5E_SCC, H5E_NOT_IMPLEMTED, FAIL,
                "direct chunk read is not currently supported or implemented");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_direct_chunk_read() */

/*-------------------------------------------------------------------------
 * Function: H5SC_direct_chunk_write
 *
 * Purpose:  Writes the chunk that starts at coordinates
 * give by offset directly from buf to disk, without any
 * decoding or conversion. First evicts that chunk from
 * cache if it is present.
 *
 * Not currently implemented or supported.
 *
 * Return:   SUCCEED on success, FAIL on failure
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_direct_chunk_write(H5SC_t *cache, H5D_t *dset, const hsize_t *offset, H5_ATTR_UNUSED void *udata,
                        const void *buf)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset->shared->layout.sc_ops);
    assert(offset);
    assert(buf);

    HGOTO_ERROR(H5E_SCC, H5E_NOT_IMPLEMTED, FAIL,
                "direct chunk write is not currently supported or implemented");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_direct_chunk_write() */

/*-------------------------------------------------------------------------
 * Function: H5SC_get_defined
 *
 * Purpose:
 *   Construct a dataset dataspace containing the elements that are both
 *     selected by file_space and currently defined in a structured sparse
 *     dataset.
 *
 *   The input selection is decomposed into logical chunks through
 *     H5SC__selection_io_info_init(). For each participating chunk, the SCC
 *     chunk key is computed and the cache is checked for an existing entry.
 *     Resident decoded chunk state is treated as authoritative so that
 *     defined values modified in memory but not yet written to disk are
 *     reflected in the result.
 *
 *   Nonresident chunks are queried without creating SCC entries or
 *     materializing full resident chunk state. H5SC__get_defined_chunk()
 *     performs the required layout lookup and, when necessary, reads and
 *     decodes only the defined-value metadata. An allocated chunk reporting
 *     defined_values_size == 0 is handled as a fast path in which all
 *     selected values in that logical chunk are considered defined.
 *
 *   Per-chunk defined-value selections are translated from logical chunk
 *     coordinates into dataset coordinates and merged into the returned
 *     dataspace.
 *
 *   The operation is intended to be cache-neutral: it does not create
 *     H5SC_chunk_t entries, alter SCC LRU ordering, pin chunks, or modify
 *     SCC byte accounting. Existing resident state may be inspected but is
 *     not otherwise modified.
 *
 *   H5S_SEL_ALL and H5S_SEL_HYPERSLABS input selections are currently
 *     supported. Point selections are intentionally unsupported by the
 *     current SCC interface.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the shared chunk cache associated with dset. Used only to
 *     locate existing dataset/chunk state; the operation does not modify
 *     cache residency or accounting.
 *
 *   H5D_t *dset:
 *     Pointer to the structured sparse dataset whose defined elements are
 *     being queried.
 *
 *   const H5S_t *file_space:
 *     Dataset dataspace containing the input selection. Only elements that
 *     are both selected here and currently defined are included in the
 *     returned dataspace.
 *
 * Return:
 *   Success:
 *     Pointer to a newly allocated dataset-sized H5S_t containing the
 *     resulting defined-value selection. Ownership is transferred to the
 *     caller.
 *
 *   Failure:
 *     NULL.
 *-------------------------------------------------------------------------
 */
H5S_t *
H5SC_get_defined(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space)
{
    H5SC_dset_header_t *dset_hdr   = NULL;
    H5SC_io_info_t     *sc_io_info = NULL;
    H5S_t              *defined    = NULL;
    hsize_t             dset_dims[H5S_MAX_RANK];
    hsize_t             chunk_dims[H5S_MAX_RANK];
    unsigned            ndims;
    bool                partial_bound_chunks_different_encoding = false;
    H5O_stc_pline_t    *pline                                   = NULL;
    hbool_t             filtered                                = false;
    H5S_t              *ret_value                               = NULL;
    haddr_t             md_tag                                  = HADDR_UNDEF;
    bool                md_tag_set                              = false;

    FUNC_ENTER_NOAPI(NULL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset->shared);
    assert(dset->shared->layout.sc_ops);
    assert(file_space);

    assert(dset->shared->layout.sc_ops->lookup);
    assert(dset->shared->layout.sc_ops->layout_query);

    ndims = dset->shared->ndims;

    if (ndims == 0 || ndims > H5S_MAX_RANK)
        HGOTO_ERROR(H5E_DATASET, H5E_BADVALUE, NULL, "invalid dataset rank for SCC get-defined operation");

    if (H5S_get_simple_extent_dims(file_space, dset_dims, NULL) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, NULL, "unable to get dataset dimensions");

    if (dset->shared->layout.sc_ops->layout_query(dset, chunk_dims, NULL,
                                                  &partial_bound_chunks_different_encoding) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, NULL, "unable to query structured chunk layout");

    pline = &(dset->shared->dcpl_cache.stc_pline);
    if (pline && pline->tot_filt_nsects)
        filtered = true;

    /*
     * Create the result with the same extent as file_space, but begin with
     * an empty selection.
     */
    if (NULL == (defined = H5S_copy(file_space, false, true)))
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, NULL, "unable to copy input dataspace");

    if (H5S_select_none(defined) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, NULL, "unable to clear defined-value result selection");

    /*
     * Reuse the dataset-owned selection scratch structure. No cache entries
     * are created or modified by the common selection initializer.
     */
    dset_hdr = H5SC__ht_dset_find(cache, dset->oloc.addr);
    if (!dset_hdr)
        HGOTO_ERROR(H5E_SCC, H5E_HT_NOTFOUND, NULL, "dataset should have been added to SCC upon creation");

    sc_io_info = dset_hdr->io_info;
    assert(sc_io_info);

    if (H5SC__selection_io_info_init(dset, file_space, sc_io_info) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_CANTINIT, NULL, "unable to initialize SCC defined-value selection");

    H5AC_tag(dset->oloc.addr, &md_tag);
    md_tag_set = true;

    for (size_t i = 0; i < sc_io_info->num_sel_chunks; i++) {
        H5SC_io_sel_chunk_t *sel = &sc_io_info->sel_chunks[i];
        H5SC_chunk_key_t     key;
        H5SC_chunk_t        *cached_chk;
        H5S_t               *chunk_defined = NULL;

        if (H5SC__compute_chunk_key(&dset->oloc.addr, &sel->chk_log_coord, &key) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CHUNK_KEY, NULL, "unable to compute SCC chunk key");

        cached_chk = H5SC__ht_chunk_find(cache, &key);

        if (H5SC__get_defined_chunk(dset, sel, cached_chk, filtered, partial_bound_chunks_different_encoding,
                                    &chunk_defined) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTGET_SELECTION, NULL,
                        "unable to determine defined values for selected chunk");

        if (chunk_defined) {
            if (H5S_GET_SELECT_NPOINTS(chunk_defined) > 0)
                if (H5SC__merge_defined_chunk_selection(defined, chunk_defined, sel, ndims, dset_dims,
                                                        chunk_dims) < 0)
                    HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, NULL,
                                "unable to merge defined-value selection");

            if (H5S_close(chunk_defined) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, NULL,
                            "unable to close chunk defined-value selection");
        }
    }

    ret_value = defined;
    defined   = NULL;

done:
    if (md_tag_set)
        H5AC_tag(md_tag, NULL);

    /*
     * Selection decomposition uses reusable dataset-owned scratch state.
     * Reset it regardless of success or failure.
     */
    if (sc_io_info) {
        if (H5SC__io_info_reset(sc_io_info) < 0)
            HDONE_ERROR(H5E_SCC, H5E_CANTRELEASE, NULL, "unable to reset get-defined selection information");
    }

    if (defined) {
        if (H5S_close(defined) < 0)
            HDONE_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, NULL, "unable to release defined-value result");
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_get_defined() */

/*-------------------------------------------------------------------------
 * Function:    H5SC_erase
 *
 * Purpose:     Erase the selected elements from a structured-chunk dataset
 *              managed by the shared chunk cache.
 *
 *              The supplied FILE_SPACE selection is decomposed into
 *              chunk-local selections by H5SC__erase_io_info_init(). For
 *              each selected chunk, this routine ensures that a resident
 *              decoded chunk object is available, restricts the erase to
 *              currently defined values when the layout provides a
 *              defined_values callback, and invokes sc_ops->erase_values().
 *
 *              Chunks that become empty are removed immediately from
 *              structured storage, detached from the dataset chunk LRU,
 *              removed from the SCC chunk hash table, and released. Chunks
 *              that retain defined values remain resident in the SCC. Their
 *              resident-size accounting is reconciled, their dataset-local
 *              resident-size history is updated, and they are marked dirty
 *              for persistence by a subsequent SCC flush or flush-and-evict
 *              operation.
 *
 *              The routine preserves SCC pin/unpin accounting for chunks
 *              selected during erase initialization and reconciles
 *              dataset/cache size accounting after the erase operation
 *              completes.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */

herr_t
H5SC_erase(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space)
{
    H5SC_dset_header_t  *dset_hdr      = NULL;
    H5SC_io_info_t      *sc_io_info    = NULL;
    H5SC_io_sel_chunk_t *base          = NULL;
    size_t               sel_start     = 0;
    size_t               sel_count     = 0;
    size_t               old_dset_size = 0;
    haddr_t              md_tag        = HADDR_UNDEF;
    bool                 md_tag_set    = false;
    herr_t               ret_value     = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset->shared->layout.sc_ops);
    assert(file_space);

    /* Check for support for erasing values */
    if (!dset->shared->layout.sc_ops->erase_values)
        HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL, "dataset does not support erasing values");

    dset_hdr = H5SC__ht_dset_find(cache, dset->oloc.addr);
    if (dset_hdr == NULL)
        HGOTO_ERROR(H5E_SCC, H5E_HT_NOTFOUND, FAIL,
                    "dataset should have been added to the hash table upon creation");

    sc_io_info = dset_hdr->io_info;
    assert(sc_io_info);

    old_dset_size = dset_hdr->curr_dset_size;

    if (H5SC__erase_io_info_init(cache, dset, file_space) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to initialize SCC erase request");

    base      = sc_io_info->sel_chunks;
    sel_start = sc_io_info->sel_start;
    sel_count = sc_io_info->num_sel_chunks;

    if (sel_start > sc_io_info->sel_chunks_alloced || sel_count > sc_io_info->sel_chunks_alloced - sel_start)
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "invalid erase selection window");

    if (sel_count == 0)
        HGOTO_DONE(SUCCEED);

    H5AC_tag(dset->oloc.addr, &md_tag);
    md_tag_set = true;

    for (size_t j = 0; j < sel_count; j++) {
        H5SC_io_sel_chunk_t *sel          = H5SC__io_sel_at(sc_io_info, j);
        H5SC_chunk_t        *chk          = sel->cached_chunk;
        H5S_t               *erase_sel    = NULL;
        bool                 delete_chunk = false;
        haddr_t              old_addr;
        hsize_t              old_disk_size;
        size_t               nbytes;
        size_t               alloc_size;

        assert(sel);
        assert(chk);

        /* Partial chunk support is not implemented yet */
        if (chk->partial_IO)
            HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL,
                        "erase for partial structured chunks is not implemented");

        /*
         * Acquire resident chunk if necessary, following the same pattern as
         * H5SC_write() / H5SC_read().
         */
        if (!chk->chunk_obj) {
            const hsize_t   *scaled_arr[1];
            haddr_t         *addr_arr[1];
            hsize_t         *size_arr[1];
            hsize_t         *def_sz_arr[1];
            size_t          *size_hint_arr[1];
            size_t          *def_sz_hint_arr[1];
            void            *udata_arr[1]                            = {NULL};
            haddr_t          addr_local                              = HADDR_UNDEF;
            hsize_t          size_local                              = 0;
            hsize_t          encoded_size_local                      = 0;
            hsize_t          def_local                               = 0;
            size_t           hint_local                              = 0;
            size_t           nbytes_local                            = 0;
            size_t           def_hint_local                          = 0;
            bool             partial_bound                           = false;
            bool             partial_bound_chunks_different_encoding = false;
            H5O_stc_pline_t *pline                                   = NULL;
            hbool_t          filtered                                = false;
            void            *chunk_buf                               = NULL;

            scaled_arr[0]      = chk->scaled;
            addr_arr[0]        = &addr_local;
            size_arr[0]        = &size_local;
            def_sz_arr[0]      = &def_local;
            size_hint_arr[0]   = &hint_local;
            def_sz_hint_arr[0] = &def_hint_local;

            /* If the dataset has never allocated structured-chunk storage, then there
             * can be no defined on-disk chunk here. Erasing is a no-op.
             */
            if (!(*dset->shared->layout.ops->is_space_alloc)(&dset->shared->layout.storage)) {

                if (!sel->pin_held || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                "unable to release erase pin for unallocated storage");

                sel->pin_held = false;
                chk->last_op  = H5SC_TAG_UNPIN_ERASE_DONE;

                continue;
            }

            if (dset->shared->layout.sc_ops->lookup(dset, 1, scaled_arr, addr_arr, size_arr, def_sz_arr,
                                                    size_hint_arr, def_sz_hint_arr, udata_arr) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "erase: lookup callback failed");

            /* Sparse miss: no chunk exists, so nothing is defined here to erase */
            if (!H5_addr_defined(addr_local)) {
                if (udata_arr[0])
                    udata_arr[0] = H5MM_xfree(udata_arr[0]);

                if (!sel->pin_held || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                "unable to release erase pin after sparse miss");

                sel->pin_held = false;
                chk->last_op  = H5SC_TAG_UNPIN_ERASE_DONE;

                continue;
            }

            /*
             * Preserve the actual structured-storage allocation length. decode() may
             * replace size_local with the decoded representation size, which must not be
             * passed to delete_chunk() as the on-disk block length.
             */
            encoded_size_local = size_local;

            if (hint_local == 0)
                HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                            "erase: invalid size_hint for decoded chunk allocation");

            if (dset->shared->layout.sc_ops->layout_query(dset, NULL, NULL,
                                                          &partial_bound_chunks_different_encoding) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query partial-bound encoding policy");

            pline = &(dset->shared->dcpl_cache.stc_pline);
            if (pline && pline->tot_filt_nsects)
                filtered = true;

            if (filtered && partial_bound_chunks_different_encoding &&
                H5D__chunk_is_partial_edge_chunk(dset->shared->ndims, dset->shared->layout.u.struct_chunk.dim,
                                                 chk->scaled, dset->shared->curr_dims))
                partial_bound = true;

            if (NULL == (chunk_buf = H5MM_malloc(hint_local)))
                HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                            "memory allocation failed for raw data chunk (erase)");

            if (H5F_block_read(dset->oloc.file, H5FD_MEM_DRAW, addr_local, size_local, chunk_buf) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_READERROR, FAIL, "erase: block read failed");

            H5_CHECKED_ASSIGN(nbytes_local, size_t, size_local, hsize_t);

            if (dset->shared->layout.sc_ops->decode(dset, &nbytes_local, &hint_local, partial_bound,
                                                    &chunk_buf, udata_arr[0]) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "erase: decode callback failed");

            chk->chunk_obj = chunk_buf;
            chk->udata     = udata_arr[0];
            chk->disk_addr = addr_local;
            H5_CHECKED_ASSIGN(chk->disk_nbytes, size_t, encoded_size_local, hsize_t);

            if (H5SC__chunk_set_dirty(cache, dset_hdr, chk, false) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "unable to mark decoded erase chunk clean");

            if (H5SC__chunk_update_cached_size(cache, dset_hdr, chk, hint_local) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "erase: chunk size accounting failed");
        }

        /*
         * Restrict erase to defined values if supported.
         * The defined_values callback returns a selection in logical chunk
         * coordinates, matching the erase_values callback contract.
         */
        erase_sel = sel->file_space;
        if (dset->shared->layout.sc_ops->defined_values) {
            H5S_t *defined_sel = NULL;

            if (dset->shared->layout.sc_ops->defined_values(dset, sel->file_space, chk->chunk_obj,
                                                            &defined_sel, chk->udata) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "erase: unable to query defined values");

            if (!defined_sel || H5S_GET_SELECT_NPOINTS(defined_sel) == 0) {
                if (defined_sel && H5S_close(defined_sel) < 0)
                    HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL,
                                "can't release defined-values dataspace");

                if (!sel->pin_held || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                "unable to release erase pin for empty defined selection");

                sel->pin_held = false;
                chk->last_op  = H5SC_TAG_UNPIN_ERASE_DONE;

                continue;
            }

            erase_sel = defined_sel;
        }

        old_addr      = chk->disk_addr;
        old_disk_size = (hsize_t)chk->disk_nbytes;
        nbytes        = chk->disk_nbytes;
        alloc_size    = chk->cached_chunk_size;

        if (dset->shared->layout.sc_ops->erase_values(dset, erase_sel, &nbytes, &alloc_size, chk->chunk_obj,
                                                      &delete_chunk, chk->udata) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTREMOVE, FAIL, "erase: erase_values callback failed");

        if (erase_sel != sel->file_space) {
            if (H5S_close(erase_sel) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CANTRELEASE, FAIL, "can't release temporary erase selection");
            erase_sel = NULL;
        }

        if (delete_chunk) {

            if (H5_addr_defined(old_addr)) {
                if (dset->shared->layout.sc_ops->delete_chunk(dset, chk->scaled, old_addr, old_disk_size) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTDELETE, FAIL, "erase: unable to delete empty chunk");
            }

            if (H5SC__chunk_lru_remove(cache, dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL, "failed to remove chunk from dataset LRU");

            if (H5SC__chunk_teardown_resident(dset, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTRELEASE, FAIL,
                            "failed to tear down resident chunk state during erase");

            if (H5SC__ht_chunk_delete(cache, &chk->data_key) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL,
                            "failed to remove chunk from hash table during erase");

            /*
             * The selected chunk is being destroyed while its request pin is still
             * logically owned. LRU removal sees the chunk as pinned and therefore removes
             * no reclaimable contribution. Clear selection ownership before freeing the
             * chunk; no unpin transition is needed because the object no longer exists.
             */
            sel->pin_held     = false;
            sel->cached_chunk = NULL;

            chk = H5MM_xfree(chk);
            continue;
        }

        /* Surviving chunk remains resident and dirty until flush or eviction. */
        if (H5SC__chunk_update_cached_size(cache, dset_hdr, chk, alloc_size) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                        "erase: chunk size accounting failed after erase_values");

        H5SC__update_resident_estimate_history(dset_hdr, alloc_size);

        if (H5SC__chunk_set_dirty(cache, dset_hdr, chk, true) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "unable to mark erase chunk dirty");

        if (!sel->pin_held || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "unable to release erase request pin");

        sel->pin_held = false;
        chk->last_op  = H5SC_TAG_UNPIN_ERASE_DONE;
    }

    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "global size accounting failed (SCC erase)");

#if H5SC_DO_SANITY_CHECKS
    if (H5SC__verify_cached_reclaimability(cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "cached reclaimability mismatch after SCC erase");
#endif

done:
    if (md_tag_set)
        H5AC_tag(md_tag, NULL);

    if (ret_value < 0 && sc_io_info && base && sel_count > 0) {
        for (size_t j = 0; j < sel_count; j++) {
            H5SC_io_sel_chunk_t *sel = H5SC__io_sel_at(sc_io_info, j);
            H5SC_chunk_t        *chk = sel->cached_chunk;

            if (sel->pin_held) {
                if (!chk || H5SC__chunk_unpin(cache, dset_hdr, chk) < 0)
                    HDONE_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "unable to unwind erase request pin");
                else {
                    sel->pin_held = false;
                    chk->last_op  = H5SC_TAG_UNPIN_ERASE_DONE_ERROR;
                }
            }
        }
    }

    if (sc_io_info) {
        if (H5SC__io_info_reset(sc_io_info) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't close erase I/O info");
    }

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_erase() */

/*-------------------------------------------------------------------------
 * Function:    H5SC__chunk_outside_extent
 *
 * Purpose:     Determine whether a scaled chunk coordinate lies completely
 *              outside the dataset's current extent.
 *
 *              A chunk is outside the extent if its starting element
 *              coordinate is greater than or equal to the current dataset
 *              extent in any dimension. Such a chunk has no valid in-extent
 *              elements and may be deleted during extent-prune processing.
 *
 * Return:      true if the chunk is outside the current extent;
 *              false otherwise.
 *
 *-------------------------------------------------------------------------
 */

static bool
H5SC__chunk_outside_extent(const H5D_t *dset, const hsize_t *chunk_dims, const hsize_t *scaled)
{
    const hsize_t *new_dims = dset->shared->curr_dims;
    unsigned       u;

    assert(dset);
    assert(chunk_dims);
    assert(scaled);

    for (u = 0; u < dset->shared->ndims; u++) {
        hsize_t chunk_start = scaled[u] * chunk_dims[u];

        if (chunk_start >= new_dims[u])
            return true;
    }

    return false;
} /* end H5SC__chunk_outside_extent() */

/*-------------------------------------------------------------------------
 * Function:    H5SC__chunk_is_partial_bound
 *
 * Purpose:     Determine whether a scaled chunk coordinate represents a
 *              partial-bound chunk with respect to the supplied dataset
 *              dimensions.
 *
 *              A chunk is partial-bound if it starts before the dataset bound
 *              in a dimension but extends past that bound. Such chunks may
 *              require different structured-chunk encoding or re-encoding
 *              when the dataset extent changes.
 *
 * Return:      true if the chunk is partial-bound;
 *              false otherwise.
 *
 *-------------------------------------------------------------------------
 */

static bool
H5SC__chunk_is_partial_bound(unsigned ndims, const hsize_t *chunk_dims, const hsize_t *scaled,
                             const hsize_t *dims)
{
    unsigned u;

    assert(chunk_dims);
    assert(scaled);
    assert(dims);

    for (u = 0; u < ndims; u++) {
        hsize_t chunk_start = scaled[u] * chunk_dims[u];
        hsize_t chunk_end   = chunk_start + chunk_dims[u];

        if (chunk_start < dims[u] && chunk_end > dims[u])
            return true;
    }

    return false;
} /* end H5SC__chunk_is_partial_bound() */

/*-------------------------------------------------------------------------
 * Function:    H5SC__chunk_needs_erase
 *
 * Purpose:     Determine whether a surviving chunk overlaps values that
 *              became invalid as a result of an extent shrink.
 *
 *              This helper identifies chunks that still intersect the new
 *              dataset extent but also overlap the region that was valid
 *              under OLD_DIMS and is now outside the current extent. Such
 *              chunks must have their newly invalid chunk-local values erased
 *              rather than being deleted wholesale.
 *
 * Return:      true if the chunk needs invalid-region erase processing;
 *              false otherwise.
 *
 *-------------------------------------------------------------------------
 */

static bool
H5SC__chunk_needs_erase(const H5D_t *dset, const hsize_t *chunk_dims, const hsize_t *scaled,
                        const hsize_t *old_dims)

{
    const hsize_t *new_dims = dset->shared->curr_dims;
    unsigned       u;

    assert(dset);
    assert(chunk_dims);
    assert(scaled);
    assert(old_dims);

    for (u = 0; u < dset->shared->ndims; u++) {
        hsize_t chunk_start = scaled[u] * chunk_dims[u];
        hsize_t chunk_end   = chunk_start + chunk_dims[u];

        if (new_dims[u] < old_dims[u]) {
            if (chunk_start < new_dims[u] && chunk_end > new_dims[u])
                return true;
        }
    }

    return false;
} /* end H5SC__chunk_needs_erase() */

/*-------------------------------------------------------------------------
 * Function:    H5SC__erase_chunk_invalid_extent_region
 *
 * Purpose:     Erase newly invalid values from a resident structured chunk
 *              after a dataset extent shrink.
 *
 *              The helper computes the portion of CHK that was valid before
 *              the shrink using OLD_DIMS and the portion that remains valid
 *              under the dataset's current extent. It then erases the
 *              difference between those regions using one or more
 *              non-overlapping chunk-local hyperslab selections.
 *
 *              The erase is intentionally issued as simple slabs rather than
 *              a composite OR/NOT selection because the structured-chunk
 *              erase path is sensitive to complex hyperslab selections in
 *              partial-bound cases.
 *
 *              If sc_ops->erase_values() reports that the chunk should be
 *              deleted, this helper preserves the chunk when it still has a
 *              non-empty geometric intersection with the new extent. This
 *              prevents extent pruning from deleting a partial-bound survivor
 *              whose invalid suffix was erased.
 *
 *              On return, NBYTES and ALLOC_SIZE contain the updated encoded
 *              and decoded/resident sizes reported by erase_values().
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__erase_chunk_invalid_extent_region(H5D_t *dset, const hsize_t *chunk_dims, const hsize_t *old_dims,
                                        H5SC_chunk_t *chk, size_t *nbytes, size_t *alloc_size,
                                        bool *delete_chunk)
{
    hsize_t origin[H5S_MAX_RANK];
    hsize_t old_valid_count[H5S_MAX_RANK];
    hsize_t new_valid_count[H5S_MAX_RANK];
    bool    has_invalid          = false;
    bool    has_new_valid_region = true;
    herr_t  ret_value            = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(chunk_dims);
    assert(old_dims);
    assert(chk);
    assert(nbytes);
    assert(alloc_size);
    assert(delete_chunk);
    assert(chk->chunk_obj);
    assert(dset->shared->layout.sc_ops);
    assert(dset->shared->layout.sc_ops->erase_values);

    *delete_chunk = false;

    /*
     * Compute the portion of this chunk that was valid before the shrink and
     * the portion that remains valid after the shrink, both in chunk-local
     * coordinates.
     */
    for (unsigned u = 0; u < dset->shared->ndims; u++) {
        origin[u] = chk->scaled[u] * chunk_dims[u];

        if (origin[u] >= old_dims[u])
            old_valid_count[u] = 0;
        else {
            hsize_t old_remaining = old_dims[u] - origin[u];
            old_valid_count[u]    = MIN(chunk_dims[u], old_remaining);
        }

        if (origin[u] >= dset->shared->curr_dims[u])
            new_valid_count[u] = 0;
        else {
            hsize_t new_remaining = dset->shared->curr_dims[u] - origin[u];
            new_valid_count[u]    = MIN(chunk_dims[u], new_remaining);
        }

        if (new_valid_count[u] < old_valid_count[u])
            has_invalid = true;

        if (new_valid_count[u] == 0)
            has_new_valid_region = false;
    }

    if (!has_invalid)
        HGOTO_DONE(SUCCEED);

    /*
     * Erase old_valid - new_valid as a union of non-overlapping simple slabs.
     *
     * For dimension u, erase the suffix newly outside the extent in that
     * dimension while restricting all earlier dimensions to their new-valid
     * prefix and all later dimensions to their old-valid extent.
     */
    for (unsigned u = 0; u < dset->shared->ndims; u++) {
        H5S_t  *erase_space = NULL;
        hsize_t start[H5S_MAX_RANK];
        hsize_t count[H5S_MAX_RANK];
        hsize_t npoints = 0;
        bool    empty   = false;

        if (new_valid_count[u] >= old_valid_count[u])
            continue;

#if H5SC_DO_SANITY_CHECKS
        assert(dset->shared->ndims > 0);
        assert(dset->shared->ndims <= H5S_MAX_RANK);
#endif

        if (NULL == (erase_space = H5S_create_simple(dset->shared->ndims, chunk_dims, NULL)))
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCREATE, FAIL, "unable to create chunk-local erase dataspace");

        if (H5S_select_none(erase_space) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, FAIL, "unable to clear chunk-local erase selection");

        for (unsigned v = 0; v < dset->shared->ndims; v++) {
            start[v] = 0;

            if (v < u)
                count[v] = new_valid_count[v];
            else
                count[v] = old_valid_count[v];
        }

        start[u] = new_valid_count[u];
        count[u] = old_valid_count[u] - new_valid_count[u];

        for (unsigned v = 0; v < dset->shared->ndims; v++)
            if (count[v] == 0)
                empty = true;

        if (empty) {
            if (H5S_close(erase_space) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CLOSEERROR, FAIL, "unable to close empty erase dataspace");
            continue;
        }

        if (H5S_select_hyperslab(erase_space, H5S_SELECT_SET, start, NULL, count, NULL) < 0) {
            if (H5S_close(erase_space) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CLOSEERROR, FAIL,
                            "unable to close erase dataspace after selection failure");
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, FAIL,
                        "unable to select invalid chunk-local erase slab");
        }

        npoints = H5S_GET_SELECT_NPOINTS(erase_space);
        if (npoints == 0) {
            if (H5S_close(erase_space) < 0)
                HGOTO_ERROR(H5E_DATASPACE, H5E_CLOSEERROR, FAIL,
                            "unable to close erase dataspace after npoints validation");
            erase_space = NULL;
            continue;
        }

        if (npoints > 0) {
            if ((dset->shared->layout.sc_ops->erase_values)(dset, erase_space, nbytes, alloc_size,
                                                            chk->chunk_obj, delete_chunk, chk->udata) < 0) {
                if (H5S_close(erase_space) < 0)
                    HGOTO_ERROR(H5E_DATASPACE, H5E_CLOSEERROR, FAIL,
                                "unable to close erase dataspace after erase failure");
                HGOTO_ERROR(H5E_DATASET, H5E_CANTREMOVE, FAIL,
                            "unable to erase out-of-bounds values from structured chunk");
            }

            /*
             * Extent pruning has geometric knowledge that erase_values()
             * does not: if this chunk still intersects the new extent, it
             * must survive even if the erased invalid suffix causes the
             * lower-level erase helper to report delete_chunk.
             */
            if (has_new_valid_region)
                *delete_chunk = false;
        }

        if (H5S_close(erase_space) < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CLOSEERROR, FAIL, "unable to close erase dataspace");

        if (*delete_chunk)
            HGOTO_DONE(SUCCEED);
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__erase_chunk_invalid_extent_region() */

/*-------------------------------------------------------------------------
 * Function:    H5SC__prune_one_scaled_coord_outside_extent
 *
 * Purpose:     Delete one structured-storage chunk coordinate that lies
 *              outside the dataset's new extent.
 *
 *              This helper is used by Pass 2 of H5SC_prune_by_extent() to
 *              clean up whole chunks that may exist in structured storage but
 *              are no longer represented by SCC-resident chunk objects.
 *
 *              The helper performs a storage lookup for the scaled chunk
 *              coordinate and, if the chunk exists, deletes it through the
 *              layout's delete_chunk callback. It does not allocate or insert
 *              an H5SC_chunk_t and does not update per-chunk SCC state.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__prune_one_scaled_coord_outside_extent(H5SC_t *cache, H5D_t *dset, const hsize_t *scaled)
{
    haddr_t addr      = HADDR_UNDEF;
    hsize_t nbytes    = 0;
    bool    exists    = false;
    herr_t  ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(scaled);
    assert(dset->shared->layout.sc_ops);
    assert(dset->shared->layout.sc_ops->delete_chunk);

    if (H5SC__lookup_chunk_for_prune(dset, scaled, &addr, &nbytes, &exists) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to look up out-of-extent structured chunk");

    if (!exists) {
        HGOTO_DONE(SUCCEED);
    }

    if ((dset->shared->layout.sc_ops->delete_chunk)(dset, scaled, addr, nbytes) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTDELETE, FAIL, "unable to delete out-of-extent structured chunk");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__prune_one_scaled_coord_outside_extent() */

/*-------------------------------------------------------------------------
 * Function:    H5SC__for_each_scaled_coord_outside_extent
 *
 * Purpose:     Iterate over scaled chunk coordinates that were present in
 *              the old chunk grid but are outside the new chunk grid after
 *              an extent shrink.
 *
 *              Each outside coordinate is passed to
 *              H5SC__prune_one_scaled_coord_outside_extent(), which removes
 *              any corresponding storage-only structured chunk.
 *
 *              Cached chunks are handled earlier by Pass 1 of
 *              H5SC_prune_by_extent(). This helper is therefore limited to
 *              storage-only whole-chunk cleanup and intentionally avoids SCC
 *              chunk creation, insertion, or per-chunk accounting.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__for_each_scaled_coord_outside_extent(H5SC_t *cache, H5D_t *dset, unsigned ndims,
                                           const hsize_t *old_nchunks, const hsize_t *new_nchunks)
{
    hsize_t scaled[H5S_MAX_RANK];
    herr_t  ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(ndims > 0);
    assert(old_nchunks);
    assert(new_nchunks);

    memset(scaled, 0, sizeof(scaled));

    while (true) {
        bool outside = false;

        for (unsigned u = 0; u < ndims; u++) {

            if (new_nchunks[u] < old_nchunks[u] && scaled[u] >= new_nchunks[u]) {
                outside = true;
                break;
            }
        }

        if (outside) {

            for (unsigned u = 0; u < ndims; u++)

                if (H5SC__prune_one_scaled_coord_outside_extent(cache, dset, scaled) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTDELETE, FAIL,
                                "unable to prune out-of-extent structured chunk");
        }

        for (int u = (int)ndims - 1; u >= 0; u--) {
            scaled[u]++;

            if (scaled[u] < old_nchunks[u])
                break;

            scaled[u] = 0;

            if (u == 0)
                HGOTO_DONE(SUCCEED);
        }
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__for_each_scaled_coord_outside_extent() */

/*-------------------------------------------------------------------------
 * Function:    H5SC__lookup_chunk_for_prune
 *
 * Purpose:     Query structured-chunk storage for a chunk considered during
 *              extent-prune cleanup.
 *
 *              This helper calls the layout lookup callback for a single
 *              scaled chunk coordinate and reports whether an allocated
 *              structured chunk exists. If present, the chunk's disk address
 *              and encoded size are returned for use by delete_chunk().
 *
 *              The lookup callback may allocate temporary udata. This helper
 *              owns that temporary udata and releases it before returning.
 *              No H5SC_chunk_t is created or inserted.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__lookup_chunk_for_prune(H5D_t *dset, const hsize_t *scaled, haddr_t *addr, hsize_t *disk_nbytes,
                             bool *exists)
{
    const hsize_t *scaled_arr[1];
    haddr_t       *addr_arr[1];
    hsize_t       *size_arr[1];
    hsize_t        defined_values_size = 0;
    hsize_t       *defined_values_size_arr[1];
    size_t         size_hint = 0;
    size_t        *size_hint_arr[1];
    size_t         defined_values_size_hint = 0;
    size_t        *defined_values_size_hint_arr[1];
    void          *udata_arr[1] = {NULL};
    herr_t         ret_value    = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(dset->shared);
    assert(dset->shared->layout.sc_ops);
    assert(dset->shared->layout.sc_ops->lookup);
    assert(scaled);
    assert(addr);
    assert(disk_nbytes);
    assert(exists);

    *addr        = HADDR_UNDEF;
    *disk_nbytes = 0;
    *exists      = false;

    scaled_arr[0]                   = scaled;
    addr_arr[0]                     = addr;
    size_arr[0]                     = disk_nbytes;
    defined_values_size_arr[0]      = &defined_values_size;
    size_hint_arr[0]                = &size_hint;
    defined_values_size_hint_arr[0] = &defined_values_size_hint;

    if ((dset->shared->layout.sc_ops->lookup)(dset, (size_t)1, scaled_arr, addr_arr, size_arr,
                                              defined_values_size_arr, size_hint_arr,
                                              defined_values_size_hint_arr, udata_arr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_NOTFOUND, FAIL, "unable to look up structured chunk for extent prune");

    if (H5_addr_defined(*addr))
        *exists = true;

done:
    if (udata_arr[0])
        udata_arr[0] = H5MM_xfree(udata_arr[0]);

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__lookup_chunk_for_prune() */

/*-------------------------------------------------------------------------
 * Function:    H5SC__rekey_dset_chunks_after_extent_change
 *
 * Purpose:     Recompute logical chunk coordinates and SCC hash keys for all
 *              live chunks in a dataset after an extent change that changes
 *              the dataset's chunk grid.
 *
 *              Because multidimensional logical chunk indices can change
 *              when the chunk-grid dimensions change, this helper performs
 *              rekeying as a dataset-wide operation. It first removes all
 *              live dataset chunks from the SCC chunk hash table, then
 *              recomputes each chunk's logical coordinate and data key using
 *              the current dataset extent, and finally reinserts all chunks
 *              under their new keys.
 *
 *              Dataset LRU ordering and residency are unchanged.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__rekey_dset_chunks_after_extent_change(H5SC_t *cache, H5D_t *dset, H5SC_dset_header_t *dset_hdr,
                                            const hsize_t *chunk_dims)
{
    H5SC_chunk_t *chk       = NULL;
    haddr_t       dset_addr = dset->oloc.addr;
    herr_t        ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset_hdr);
    assert(chunk_dims);

    /*
     * Phase 1: remove all live chunks from the hash table under their
     * current keys. This avoids old-key/new-key collisions between surviving
     * chunks during grid remapping.
     */
    for (chk = dset_hdr->lru_head_ptr; chk; chk = chk->next_ptr) {
        if (H5SC__ht_chunk_delete(cache, &chk->data_key) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTREMOVE, FAIL,
                        "unable to remove dataset chunk from hash table before dataset rekey");
    }

    /*
     * Phase 2: recompute logical coordinates and keys using the current
     * post-extent-change dataset dimensions.
     */
    for (chk = dset_hdr->lru_head_ptr; chk; chk = chk->next_ptr) {
        hsize_t          elem_coord[H5S_MAX_RANK];
        hsize_t          new_log_chk_coord = 0;
        H5SC_chunk_key_t new_key;

        for (unsigned u = 0; u < dset->shared->ndims; u++) {
            elem_coord[u] = chk->scaled[u] * chunk_dims[u];

            if (elem_coord[u] >= dset->shared->curr_dims[u])
                HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                            "attempted to rekey chunk outside current dataset extent");
        }

        if (H5SC__compute_logical_chunk_index(dset->shared->ndims, dset->shared->curr_dims, chunk_dims,
                                              elem_coord, &new_log_chk_coord) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTCOMPUTE, FAIL,
                        "unable to compute logical chunk coordinate during dataset rekey");

        if (H5SC__compute_chunk_key(&dset_addr, &new_log_chk_coord, &new_key) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CHUNK_KEY, FAIL, "unable to compute chunk key during dataset rekey");

        chk->chk_log_coord = new_log_chk_coord;
        chk->data_key      = new_key;
    }

    /*
     * Phase 3: reinsert all chunks under their new keys.
     */
    for (chk = dset_hdr->lru_head_ptr; chk; chk = chk->next_ptr) {
        if (H5SC__ht_chunk_insert(cache, chk) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL,
                        "unable to reinsert dataset chunk after dataset rekey");
    }

#if H5SC_DO_SANITY_CHECKS
    if (H5SC__verify_cached_reclaimability(cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "cached reclaimability mismatch after extent change");
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__rekey_dset_chunks_after_extent_change() */

/*-------------------------------------------------------------------------
 * Function:    H5SC_prune_by_extent
 *
 * Purpose:     Reconcile SCC-managed structured chunks after a dataset extent
 *              change.
 *
 *              This routine is called after H5Dset_extent() has updated the
 *              dataset extent and structured-chunk indexing metadata. It
 *              compares the old and current extents and updates SCC state and
 *              structured storage accordingly.
 *
 *              On shrink, the routine:
 *
 *              1) Traverses cached chunks safely while allowing mutation.
 *                 - Chunks fully outside the new extent are deleted from
 *                   structured storage, evicted, removed from the dataset
 *                   LRU, and removed from the SCC chunk hash table.
 *                 - Surviving chunks that overlap newly invalid regions have
 *                   those invalid values erased through
 *                   H5SC__erase_chunk_invalid_extent_region().
 *                 - Surviving dirty or re-encoded chunks are flushed through
 *                   H5SC__flush_one_chunk().
 *
 *              2) Performs storage-only cleanup for whole chunks that are
 *                 outside the new extent but no longer represented by cached
 *                 SCC chunk objects.
 *
 *              3) Rekeys surviving chunks if the chunk-grid dimensions
 *                 changed and reconciles SCC dataset-size accounting.
 *
 *              On extent growth, the routine only rekeys live chunks when
 *              the chunk grid changes.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC_prune_by_extent(H5SC_t *cache, H5D_t *dset, const hsize_t *old_dims)
{
    H5SC_dset_header_t *dset_hdr      = NULL;
    H5SC_chunk_t       *chk           = NULL;
    H5SC_chunk_t       *next_chk      = NULL;
    size_t              old_dset_size = 0;
    hsize_t             chunk_dims[H5S_MAX_RANK];
    hsize_t             old_nchunks[H5S_MAX_RANK];
    hsize_t             new_nchunks[H5S_MAX_RANK];
    bool                partial_bound_chunks_different_encoding = false;
    herr_t              ret_value                               = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset->shared->layout.sc_ops);
    assert(old_dims);

    /* Look up dataset-local cache state. */
    dset_hdr = H5SC__ht_dset_find(cache, dset->oloc.addr);
    if (!dset_hdr)
        HGOTO_DONE(SUCCEED);

    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);

    old_dset_size = dset_hdr->curr_dset_size;

    /* Query structured chunk layout geometry and partial-bound encoding policy */
    if ((dset->shared->layout.sc_ops->layout_query)(dset, chunk_dims, NULL,
                                                    &partial_bound_chunks_different_encoding) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query structured chunk layout");

    bool extent_shrunk      = false;
    bool chunk_grid_changed = false;

    /* Compute old/new chunk-grid extents so pruning can discover chunks that
     * still exist in storage even if they are no longer represented in SCC.
     */
    for (unsigned u = 0; u < dset->shared->ndims; u++) {
        old_nchunks[u] = (old_dims[u] + chunk_dims[u] - 1) / chunk_dims[u];
        new_nchunks[u] = (dset->shared->curr_dims[u] + chunk_dims[u] - 1) / chunk_dims[u];

        if (dset->shared->curr_dims[u] < old_dims[u])
            extent_shrunk = true;

        if (old_nchunks[u] != new_nchunks[u])
            chunk_grid_changed = true;
    }

    if (!extent_shrunk) {
        if (chunk_grid_changed) {
            if (H5SC__rekey_dset_chunks_after_extent_change(cache, dset, dset_hdr, chunk_dims) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTSET, FAIL,
                            "unable to update SCC chunk keys after extent growth");
        }

        HGOTO_DONE(SUCCEED);
    }

    /*----------------------------------------------------------------------
     * Pass 1: mutation-safe traversal over cached SCC chunks
     *----------------------------------------------------------------------
     */

    for (chk = dset_hdr->lru_head_ptr; chk; chk = next_chk) {
        haddr_t old_addr;
        hsize_t old_disk_size;
        bool    old_partial;
        bool    new_partial;
        bool    encoding_changed;
        bool    delete_chunk = false;

        next_chk = chk->next_ptr;

        assert(chk->magic == H5SC_CHUNK_MAGIC);
        assert(chk->ndims == dset->shared->ndims);

        /* Partial I/O chunk state is not supported by extent pruning. */
        if (chk->partial_IO)
            HGOTO_ERROR(H5E_DATASET, H5E_UNSUPPORTED, FAIL,
                        "extent pruning for partial structured chunks is not implemented");

        old_addr      = chk->disk_addr;
        old_disk_size = (hsize_t)chk->disk_nbytes;

        old_partial = H5SC__chunk_is_partial_bound(dset->shared->ndims, chunk_dims, chk->scaled, old_dims);
        new_partial = H5SC__chunk_is_partial_bound(dset->shared->ndims, chunk_dims, chk->scaled,
                                                   dset->shared->curr_dims);

        encoding_changed = partial_bound_chunks_different_encoding && (old_partial != new_partial);

        /*--------------------------------------------------------------
         * Case 1: chunk is fully outside the new extent
         *--------------------------------------------------------------*/
        if (H5SC__chunk_outside_extent(dset, chunk_dims, chk->scaled)) {
            if (H5_addr_defined(old_addr)) {
                if ((dset->shared->layout.sc_ops->delete_chunk)(dset, chk->scaled, old_addr, old_disk_size) <
                    0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTDELETE, FAIL,
                                "unable to delete out-of-scope structured chunk");
            }

            if (chk->chunk_obj) {
                if ((dset->shared->layout.sc_ops->evict)(dset, chk->chunk_obj, chk->udata) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL,
                                "unable to evict out-of-scope structured chunk");

                chk->chunk_obj = NULL;
                chk->udata     = NULL;
            }

            if (H5SC__chunk_lru_remove(cache, dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTREMOVE, FAIL, "unable to remove chunk from dataset LRU");

            if (H5SC__ht_chunk_delete(cache, &chk->data_key) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTREMOVE, FAIL, "unable to remove chunk from chunk hash table");

            chk = H5MM_xfree(chk);

            continue;
        }

        /*--------------------------------------------------------------
         * Case 2: surviving chunk overlaps the newly invalid region
         *--------------------------------------------------------------*/
        if (H5SC__chunk_needs_erase(dset, chunk_dims, chk->scaled, old_dims)) {
            size_t nbytes     = 0;
            size_t alloc_size = 0;

            nbytes     = chk->disk_nbytes;
            alloc_size = chk->cached_chunk_size;

            if (H5SC__erase_chunk_invalid_extent_region(dset, chunk_dims, old_dims, chk, &nbytes, &alloc_size,
                                                        &delete_chunk) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTREMOVE, FAIL,
                            "unable to erase invalid extent region from structured chunk");

            if (!delete_chunk) {
                if (H5SC__chunk_update_cached_size(cache, dset_hdr, chk, alloc_size) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANTSET, FAIL,
                                "unable to update SCC chunk cached size after extent prune erase");

                H5SC__update_resident_estimate_history(dset_hdr, alloc_size);

                chk->disk_nbytes = nbytes;

                if (H5SC__chunk_set_dirty(cache, dset_hdr, chk, true) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "unable to mark extent-pruned chunk dirty");
            }
        }

        /*--------------------------------------------------------------
         * Case 3: erase emptied the chunk
         *--------------------------------------------------------------*/
        old_addr      = chk->disk_addr;
        old_disk_size = (hsize_t)chk->disk_nbytes;

        if (delete_chunk) {
            if (H5_addr_defined(old_addr)) {
                if ((dset->shared->layout.sc_ops->delete_chunk)(dset, chk->scaled, old_addr, old_disk_size) <
                    0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTDELETE, FAIL,
                                "unable to delete emptied structured chunk");
            }

            if (chk->chunk_obj) {
                if ((dset->shared->layout.sc_ops->evict)(dset, chk->chunk_obj, chk->udata) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTFREE, FAIL, "unable to evict emptied structured chunk");

                chk->chunk_obj = NULL;
                chk->udata     = NULL;
            }

            if (H5SC__chunk_lru_remove(cache, dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTREMOVE, FAIL, "unable to remove chunk from dataset LRU");

            if (H5SC__ht_chunk_delete(cache, &chk->data_key) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTREMOVE, FAIL, "unable to remove chunk from chunk hash table");

            chk = H5MM_xfree(chk);

            continue;
        }

        /*--------------------------------------------------------------
         * Case 4: surviving chunk must be re-encoded / rewritten
         *--------------------------------------------------------------*/
        if (chk->chunk_obj && (chk->dirty_flag || encoding_changed)) {
            if (H5SC__flush_one_chunk(cache, dset, dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL,
                            "unable to flush structured chunk after extent change");
        }
    }

    /*----------------------------------------------------------------------
     * Pass 2: discover whole chunks that are outside the new extent and no
     * longer represented in SCC, then delete any such chunks that still
     * exist in structured storage.
     *
     * This pass intentionally does not create H5SC_chunk_t objects and does
     * not perform per-chunk SCC accounting. Cached chunks are handled in
     * Pass 1, and SCC accounting is reconciled once at the end of this
     * routine.
     *----------------------------------------------------------------------
     */
    if (H5SC__for_each_scaled_coord_outside_extent(cache, dset, dset->shared->ndims, old_nchunks,
                                                   new_nchunks) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTDELETE, FAIL, "unable to prune storage-only out-of-extent chunks");

    if (chunk_grid_changed) {
        if (H5SC__rekey_dset_chunks_after_extent_change(cache, dset, dset_hdr, chunk_dims) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTSET, FAIL, "unable to update SCC chunk keys after extent prune");
    }

    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                    "failed to reconcile SCC accounting after extent prune");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_prune_by_extent() */

/*-------------------------------------------------------------------------
 * Function:    H5SC_set_extent_notify
 *
 * Purpose:     Notify the shared chunk cache that a structured-chunk dataset
 *              extent has changed.
 *
 *              The routine updates the dataset's cached current dimensions,
 *              refreshes structured-chunk indexing metadata, and invokes
 *              H5SC_prune_by_extent() when an allocated structured-chunk
 *              dataset has grown or shrunk.
 *
 *              Shrinks may delete out-of-extent chunks, erase newly invalid
 *              values from surviving partial-bound chunks, flush chunks that
 *              require re-encoding, and rekey chunks when the chunk grid
 *              changes. Growth may require rekeying surviving chunks when
 *              multidimensional logical chunk indices change.
 *
 * Return:      SUCCEED/FAIL
 *
 *-------------------------------------------------------------------------
 */

herr_t
H5SC_set_extent_notify(H5SC_t *cache, H5D_t *dset, const hsize_t *old_dims)
{
    hsize_t  ext_dims[H5S_MAX_RANK]; /* Extended dimension sizes */
    unsigned dim_idx;                /* Dimension index */
    bool     shrink    = false;      /* Whether any dimension shrank */
    bool     expand    = false;      /* Whether any dimension grew */
    herr_t   ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset->shared->layout.sc_ops);
    assert(old_dims);

    /* Obtain the dataset's current extent after H5Dset_extent() */
    if (H5S_get_simple_extent_dims(dset->shared->space, ext_dims, NULL) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "can't retrieve dataspace dimensions");

    /* Determine whether we shrank and/or expanded any dimensions, and update
     * the cached current dimensions.
     */
    for (dim_idx = 0; dim_idx < dset->shared->ndims; dim_idx++) {
        if (ext_dims[dim_idx] < old_dims[dim_idx])
            shrink = true;
        if (ext_dims[dim_idx] > old_dims[dim_idx])
            expand = true;

        dset->shared->curr_dims[dim_idx] = ext_dims[dim_idx];
    }

    /* Recompute structured chunk indexing metadata using the new extent */
    if (H5D_STRUCT_CHUNK == dset->shared->layout.type) {
        if (H5D__struct_chunk_set_info(dset) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTSET, FAIL, "unable to update structured chunk info");
    }

    /* Reconcile cached structured chunks with the new extent, but only if
     * storage has been allocated and the extent actually changed.
     */
    if (H5D_STRUCT_CHUNK == dset->shared->layout.type) {
        if ((expand || shrink) &&
            ((*dset->shared->layout.ops->is_space_alloc)(&dset->shared->layout.storage))) {
            if (H5SC_prune_by_extent(cache, dset, old_dims) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "unable to prune structured chunks");
        }
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_set_extent_notify() */

/**************************************************
 * SCC Structure Initialization + Operations
 **************************************************/

/*-------------------------------------------------------------------------
 * Function: H5SC_dset_create_header
 *
 * Purpose:
 *   Function that allocates and initializes an
 *   H5SC_dset_header_t payload node for a dataset. The node is initialized
 *   with the dataset object header address, default SCC accounting state,
 *   and reusable per-dataset I/O state. The node is not linked into any
 *   hash table or LRU structure; ownership remains with the caller.
 *
 *   The dataset's minimum retained size is initialized to 10 KiB by
 *   default. If the SCC quiescent limit is smaller than 10 KiB, the
 *   minimum retained size is reduced to 10 percent of the quiescent limit,
 *   rounded to the nearest integer value, with a minimum value of 1.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC instance whose quiescent-limit setting is used to
 *     determine the initial min_dset_size for the dataset header.
 *
 *   haddr_t addr:
 *     Dataset object header address to store in dset_addr.
 *
 * Return:
 *   Non-NULL pointer to a newly allocated H5SC_dset_header_t on success;
 *   NULL on failure.
 *-------------------------------------------------------------------------
 */

H5SC_dset_header_t *
H5SC_dset_create_header(H5SC_t *cache, haddr_t addr, size_t max_chunk_size)
{
    H5SC_dset_header_t *ret_value = NULL;
    size_t              min_size  = (size_t)(10ULL * 1024ULL);
    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    if (cache->SCC_quiescent_limit < (size_t)(10ULL * 1024ULL)) {
        /*
         * Set min_dset_size to 50% of SCC_quiescent_limit, rounded to nearest integer
         * min_size = (cache->SCC_quiescent_limit + (size_t)1) / (size_t)2;
         */
        /*
         * Set min_dset_size to 75% of SCC_quiescent_limit, round to nearest integer
         * min_size = (size_t)(((3ULL * cache->SCC_quiescent_limit) + 2ULL) / 4ULL);
         */
        min_size = (size_t)(2ULL * max_chunk_size);
        if (min_size < (size_t)1)
            min_size = (size_t)1;
    }

    H5SC_dset_header_t *dset_hdr = (H5SC_dset_header_t *)H5MM_malloc(sizeof(*dset_hdr));
    memset(dset_hdr, 0, sizeof(*dset_hdr));
    dset_hdr->dset_addr          = addr;
    dset_hdr->min_dset_size      = min_size;
    dset_hdr->curr_dset_size     = 0;
    dset_hdr->magic              = H5SC_DSET_HDR_MAGIC;
    dset_hdr->last_op            = H5SC_TAG_CREATE;
    dset_hdr->resize_in_progress = false;

    dset_hdr->io_info = (H5SC_io_info_t *)H5MM_calloc(sizeof(H5SC_io_info_t));
    if (!dset_hdr->io_info)
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "can't allocate dataset I/O info");

    /* Default values: “empty but reusable” */
    dset_hdr->io_info->sel_chunks         = NULL;
    dset_hdr->io_info->scratch            = NULL;
    dset_hdr->io_info->num_sel_chunks     = 0;
    dset_hdr->io_info->sel_chunks_alloced = 0;

    return dset_hdr;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_dset_create_header() */

/*-------------------------------------------------------------------------
 * Function: H5SC_dset_destroy_header
 *
 * Purpose:  Tear down SCC state for a dataset during H5D_close().
 *           On entry, the dataset should already be removed from the SCC
 *           dataset LRU and dataset hash table, but may still have resident chunks
 *           in its per-dataset chunk LRU if something went wrong upstream.
 *
 * Notes:
 *  - Frees any remaining resident chunks (including decoded resident objects).
 *  - Frees the dataset's persistent io_info (sel_chunks array, etc.).
 *  - Frees the dataset header itself.
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_dset_destroy_header(H5SC_t *cache, H5D_t *dset, H5SC_dset_header_t *dset_hdr)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset);
    assert(dset_hdr);
    assert(dset_hdr->io_info);
    assert(dset_hdr->io_info->num_sel_chunks == 0);

    if (dset_hdr->lru_head_ptr || dset_hdr->lru_tail_ptr || dset_hdr->chunk_lru_len != 0 ||
        dset_hdr->curr_dset_size != 0)
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTFREE, FAIL, "cannot destroy nonempty SCC dataset header");

    /* Free dataset I/O scratchpad */
    if (dset_hdr->io_info) {
        if (H5SC__io_info_term(dset_hdr->io_info) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTRELEASE, FAIL, "failed to terminate dataset io_info");

        dset_hdr->io_info = NULL;
    }

    /* Poison magic for safety */
    dset_hdr->magic = 0;

    dset_hdr = H5MM_xfree(dset_hdr);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_dset_destroy_header() */

/*-------------------------------------------------------------------------
 * Function: H5SC__make_chunk
 *
 * Purpose:
 *   Function that allocates and initializes an H5SC_chunk_t structure
 *     for use as an SCC chunk cache entry. The new chunk is initialized
 *     with the supplied chunk key and selected state fields, but is not
 *     linked into any hash table or LRU list; ownership remains with the
 *     caller.
 *
 *   In addition to copying the provided key, the function initializes
 *     cache-managed fields to their default starting state, including
 *     undefined or zero-valued on-disk metadata, NULL callback-related
 *     pointers, zero dimensions, and standard SCC initialization tags
 *     and magic values.
 *
 * Inputs:
 *   H5SC_chunk_key_t key:
 *     The 128-bit chunk key to copy into the new chunk’s data_key field.
 *
 *   size_t cached_sz:
 *     Initial value for cached_chunk_size.
 *
 *   size_t counter:
 *     Initial value for chunk_counter.
 *
 *   bool pio:
 *     Initial value for partial_IO, indicating whether the chunk is
 *     initially associated with partial I/O state.
 *
 * Return:
 *   Non-NULL pointer to a newly allocated and initialized H5SC_chunk_t
 *   on success;
 *   NULL on failure.
 *-------------------------------------------------------------------------
 */

H5SC_chunk_t *
H5SC__make_chunk(H5SC_chunk_key_t key, size_t cached_sz, size_t counter, bool pio)
{
    H5SC_chunk_t *chk = NULL;

    /* Allocate chunk structure */
    chk = (H5SC_chunk_t *)H5MM_calloc(sizeof(*chk));

    assert(chk);

    /* Initialize key */
    chk->data_key      = key;
    chk->chk_log_coord = 0;

    /* Callback-/payload-related fields */
    chk->chunk_obj = NULL;
    chk->udata     = NULL;

    /* Coordinate/state fields */
    chk->ndims = 0;
    /* chk->scaled[] is zero-initialized by calloc */

    /* On-disk metadata */
    chk->disk_addr   = HADDR_UNDEF;
    chk->disk_nbytes = 0;

    /* Cache bookkeeping/state */
    chk->cached_chunk_size = cached_sz;
    chk->chunk_counter     = counter;
    chk->magic             = H5SC_CHUNK_MAGIC;
    chk->last_op           = H5SC_TAG_CREATE;
    chk->dirty_flag        = false;
    chk->partial_IO        = pio;

    if (!chk)
        H5MM_free(chk);

    return chk;
} /* end H5SC__make_chunk() */

/*-------------------------------------------------------------------------
 * Function: H5SC_dset_is_empty
 *
 * Purpose:
 *   Determine whether a dataset’s per-dataset chunk LRU
 * list is completely empty.  This helper exists so external
 * callers—who only see the dataset header as an opaque
 * handle—can perform a post-flush sanity check without
 *   relying on internal struct layout or field access.
 *
 * Inputs:
 *   const H5SC_dset_header_t *dset_hdr:
 *     Pointer to a dataset header previously created by the
 * SCC.  Must not be NULL.  The function assumes the header
 * is valid and owned by the SCC.
 *
 *   hbool_t *is_empty:
 *     Output parameter.  On successful return, set to TRUE
 * if the dataset’s chunk_lru_len is zero and both the head
 * and tail chunk LRU pointers are NULL; set to FALSE
 * otherwise.  Must be non-NULL.
 *
 * Return:
 *   SUCCEED if the query completes successfully;
 *   FAIL on invalid arguments or inconsistent internal
 * state.
 *-------------------------------------------------------------------------
 * */

herr_t
H5SC_dset_is_empty(H5SC_dset_header_t *dset_hdr, bool *is_empty)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(dset_hdr);

    *is_empty =
        (dset_hdr->chunk_lru_len == 0 && dset_hdr->lru_head_ptr == NULL && dset_hdr->lru_tail_ptr == NULL);

    if (!dset_hdr)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "invalid dataset header pointer");

    if (!is_empty)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "invalid output pointer");

done:
    FUNC_LEAVE_NOAPI(ret_value);
}

/*-------------------------------------------------------------------------
 * Function: H5SC_dset_get_addr
 *
 * Purpose:
 *   Retrieve the dataset’s object header address
 * (dset_addr) from a dataset header node.  This helper
 * isolates callers from the internal definition of
 * H5SC_dset_header_t, which may be opaque outside the SCC
 * module.
 *
 * Inputs:
 *   const H5SC_dset_header_t *dset_hdr:
 *     Pointer to a valid dataset header.  The header must
 * have been created and initialized by SCC routines.  Must
 * not be NULL.
 *
 * Return:
 *   The dataset’s object header address (haddr_t) stored in
 * the header. If dset_hdr is NULL, the behavior is undefined;
 * callers should validate arguments before invoking this
 * function.
 *-------------------------------------------------------------------------
 * */

haddr_t
H5SC_dset_get_addr(H5SC_dset_header_t *dset_hdr)
{
    assert(dset_hdr);
    return dset_hdr->dset_addr;
}

/**************************************************
 * Chunk Key Functions
 **************************************************/

/*-------------------------------------------------------------------------
 * Function: H5SC__compute_chunk_key
 *
 * Purpose:
 *   Function that computes the 128-bit SCC chunk key for a chunk using
 *     the dataset object header address and the chunk’s logical chunk
 *     coordinate. The resulting key is stored in the supplied
 *     H5SC_chunk_key_t structure and is used for chunk lookup and
 *     identification within the SCC chunk hash table.
 *
 *   The key is formed by interleaving the bits of the dataset object
 *     header address and the logical chunk coordinate in an LSB-first
 *     manner. For each bit position i in [0, 63], bit i of the dataset
 *     address is written to output position 2*i and bit i of the logical
 *     chunk coordinate is written to output position 2*i + 1. This
 *     produces a deterministic 128-bit key that uniquely encodes the
 *     dataset/chunk pair within the supported bit range.
 *
 * Inputs:
 *   haddr_t *dset_object_header_addr:
 *     Pointer to the dataset object header address used as one component
 *     of the chunk key.
 *
 *   hsize_t *log_chk_coord:
 *     Pointer to the logical chunk coordinate used as the second
 *     component of the chunk key.
 *
 *   H5SC_chunk_key_t *chunk_key:
 *     Pointer to the chunk key structure to populate. The structure is
 *     reset by this function before the computed key is written.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__compute_chunk_key(const haddr_t *dset_object_header_addr /*in*/, const hsize_t *log_chk_coord /*in*/,
                        H5SC_chunk_key_t *chunk_key /*out*/)
{
    uint64_t addr;
    uint64_t addr_high;
    uint64_t addr_low;
    uint64_t log_chk;
    uint64_t log_chk_high;
    uint64_t log_chk_low;

    assert(dset_object_header_addr);
    assert(log_chk_coord);
    assert(chunk_key);

    /* Normalize the inputs to 64-bit values. */
    addr    = (uint64_t)*dset_object_header_addr;
    log_chk = (uint64_t)*log_chk_coord;

    /*
     * Separate the lower and upper 32-bit portions of each input.
     *
     * The lower portions are expanded into chunk_key->low_half, while
     * the upper portions are expanded into chunk_key->high_half.
     */
    addr_low     = addr & UINT64_C(0x00000000FFFFFFFF);
    log_chk_low  = log_chk & UINT64_C(0x00000000FFFFFFFF);
    addr_high    = addr >> 32;
    log_chk_high = log_chk >> 32;

    /*
     * Spread the lower 32 address bits into the even-numbered bit
     * positions of a 64-bit value.
     */
    addr_low = (addr_low | (addr_low << 16)) & UINT64_C(0x0000FFFF0000FFFF);
    addr_low = (addr_low | (addr_low << 8)) & UINT64_C(0x00FF00FF00FF00FF);
    addr_low = (addr_low | (addr_low << 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    addr_low = (addr_low | (addr_low << 2)) & UINT64_C(0x3333333333333333);
    addr_low = (addr_low | (addr_low << 1)) & UINT64_C(0x5555555555555555);

    /*
     * Spread the lower 32 logical-coordinate bits into the
     * even-numbered bit positions of a 64-bit value.
     */
    log_chk_low = (log_chk_low | (log_chk_low << 16)) & UINT64_C(0x0000FFFF0000FFFF);
    log_chk_low = (log_chk_low | (log_chk_low << 8)) & UINT64_C(0x00FF00FF00FF00FF);
    log_chk_low = (log_chk_low | (log_chk_low << 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    log_chk_low = (log_chk_low | (log_chk_low << 2)) & UINT64_C(0x3333333333333333);
    log_chk_low = (log_chk_low | (log_chk_low << 1)) & UINT64_C(0x5555555555555555);

    /*
     * Spread the upper 32 address bits into the even-numbered bit
     * positions of a 64-bit value.
     */
    addr_high = (addr_high | (addr_high << 16)) & UINT64_C(0x0000FFFF0000FFFF);
    addr_high = (addr_high | (addr_high << 8)) & UINT64_C(0x00FF00FF00FF00FF);
    addr_high = (addr_high | (addr_high << 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    addr_high = (addr_high | (addr_high << 2)) & UINT64_C(0x3333333333333333);
    addr_high = (addr_high | (addr_high << 1)) & UINT64_C(0x5555555555555555);

    /*
     * Spread the upper 32 logical-coordinate bits into the
     * even-numbered bit positions of a 64-bit value.
     */
    log_chk_high = (log_chk_high | (log_chk_high << 16)) & UINT64_C(0x0000FFFF0000FFFF);
    log_chk_high = (log_chk_high | (log_chk_high << 8)) & UINT64_C(0x00FF00FF00FF00FF);
    log_chk_high = (log_chk_high | (log_chk_high << 4)) & UINT64_C(0x0F0F0F0F0F0F0F0F);
    log_chk_high = (log_chk_high | (log_chk_high << 2)) & UINT64_C(0x3333333333333333);
    log_chk_high = (log_chk_high | (log_chk_high << 1)) & UINT64_C(0x5555555555555555);

    /*
     * Address bits occupy the even-numbered positions. Shifting the
     * expanded logical-coordinate values by one places their bits in
     * the odd-numbered positions.
     */
    chunk_key->low_half  = addr_low | (log_chk_low << 1);
    chunk_key->high_half = addr_high | (log_chk_high << 1);

    return SUCCEED;
} /* end H5SC__compute_chunk_key() */

/*-------------------------------------------------------------------------
 * Function: H5SC__compute_logical_chunk_index
 *
 * Purpose:
 *   Function that computes the logical chunk index for the chunk containing
 *     a specified dataset element coordinate. The computed value is a
 *     linearized, dataset-relative chunk coordinate obtained from the
 *     dataset dimensions, chunk dimensions, and element position.
 *
 *   The function first computes the number of chunks in each dimension,
 *     derives the C-order multipliers for that chunk grid, and then uses
 *     the supplied element coordinate to determine the corresponding
 *     logical chunk index. This index uniquely identifies the containing
 *     chunk within the dataset’s chunk index space.
 *
 * Inputs:
 *   unsigned ndims:
 *     The number of dimensions in the dataset. Must be greater than 0.
 *
 *   const hsize_t *dset_dims:
 *     Pointer to an array of size ndims containing the dataset dimensions.
 *
 *   hsize_t *chunk_dims:
 *     Pointer to an array of size ndims containing the chunk dimensions.
 *     Each chunk dimension must be greater than 0 and within uint32_t
 *     range.
 *
 *   const hsize_t *elem_coord:
 *     Pointer to an array of size ndims containing the dataset element
 *     coordinate whose containing chunk is to be identified.
 *
 *   hsize_t *log_chk_idx:
 *     Pointer to the location where the computed logical chunk index will
 *     be stored.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__compute_logical_chunk_index(unsigned ndims, const hsize_t *dset_dims, const hsize_t *chunk_dims,
                                  const hsize_t *elem_coord, hsize_t *log_chk_idx)
{
    herr_t   ret_value = SUCCEED;
    hsize_t  nchunks[H5S_MAX_RANK];
    hsize_t  down[H5S_MAX_RANK];
    uint32_t chunk_dims32[H5S_MAX_RANK];

    FUNC_ENTER_PACKAGE

    assert(ndims > 0);
    assert(dset_dims && chunk_dims && elem_coord && log_chk_idx);

    /* 1) number of chunks per dim: ceil(dims/chunk) */
    for (unsigned i = 0; i < ndims; i++) {
        if (chunk_dims[i] == 0) {
            HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "chunk dim must be > 0");
        }
        if (chunk_dims[i] > UINT32_MAX) {
            HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "chunk dim exceeds uint32_t range");
        }

        chunk_dims32[i] = (uint32_t)chunk_dims[i];

        nchunks[i] = (dset_dims[i] + (hsize_t)chunk_dims[i] - 1) / (hsize_t)chunk_dims[i];
    }

    /* 2) C-order multipliers */
    H5VM_array_down(ndims, nchunks, down);

    /* 3) Linear index from element coords */

    /*
     * The coordinate supplied here is expected to be an element coordinate
     * inside the dataset extent, normally the origin of a surviving chunk.
     * Callers that are processing pruned/out-of-extent chunks must not use
     * post-shrink dimensions with those coordinates.
     */

    {
        /* Sanity Check: bounds check via scaled coords */
#if H5SC_DO_SANITY_CHECKS
        for (unsigned d = 0; d < ndims; d++) {
            hsize_t scaled_d = elem_coord[d] / (hsize_t)chunk_dims[d];
            if (elem_coord[d] >= dset_dims[d] || scaled_d >= nchunks[d]) {
                HGOTO_ERROR(H5E_DATASPACE, H5E_BADVALUE, FAIL, "coordinate out of range");
            }
        }
#endif

        *log_chk_idx = H5VM_chunk_index(ndims, elem_coord, chunk_dims32, down);
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5SC__compute_logical_chunk_index */

/**************************************************
 * SCC Doubly Linked List Functions
 **************************************************/

/******************************************************************************
 *
 * Function:    H5SC__dset_lru_prepend
 *
 * Purpose:
 *   Insert a dataset header as MRU (at the head) of the global dataset-header
 *   LRU maintained in H5SC_t.{dset_lru_head_ptr,dset_lru_tail_ptr}. Pointer
 *   splicing is performed by H5SC_DLL_PREPEND; counters are updated explicitly
 *   via the sidecar (ctr->len/ctr->bytes).
 *
 * Inputs:
 *   H5SC_t                 *cache   – SCC instance holding global header LRU
 *   H5SC_dset_header_t     *dset_hdr  – Header to insert (must not already be linked)
 *
 * Returns
 *   SUCCEED on success; FAIL on already-linked, splice, or invariant error.
 *
 * Notes
 *   - Invariants are checked when H5SC_DO_SANITY_CHECKS is enabled.
 *   - No hash-table maintenance is performed here.
 */

herr_t
H5SC__dset_lru_prepend(H5SC_t *cache, H5SC_dset_header_t *dset_hdr)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *it = NULL, *pr = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);

#if H5SC_DO_SANITY_CHECKS
    /* Membership: must not already be linked */
    if (dset_hdr->next_dset_ptr || dset_hdr->prev_dset_ptr || cache->dset_lru_head_ptr == dset_hdr)
        HGOTO_ERROR(H5E_SCC, H5E_ALREADY_LINKED, FAIL, "header already linked in global LRU");
#endif

    /* MRU splice */
    H5SC_DLL_PREPEND(dset_hdr, next_dset_ptr, prev_dset_ptr, cache->dset_lru_head_ptr,
                     cache->dset_lru_tail_ptr,
                     HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "header prepend failed"));

    /* explicit accounting */
    cache->dset_lru_len++;
    cache->SCC_quiescent_size += dset_hdr->curr_dset_size;

    /* Set the struct tag */
    dset_hdr->last_op = H5SC_TAG_LRU_TOUCH;

#if H5SC_DO_SANITY_CHECKS
    H5SC_DLL_CHECK_LINKS(it, pr, cache->dset_lru_head_ptr, cache->dset_lru_tail_ptr, next_dset_ptr,
                         prev_dset_ptr,
                         HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL, "bad header LRU linkage"));
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__dset_lru_prepend() */

/******************************************************************************
 *
 * Function:    H5SC__dset_lru_remove
 *
 * Purpose:
 *   Remove a dataset header from the global dataset-header LRU and decrement
 *   counters by dset_hdr->curr_dset_size.
 *
 * Inputs:
 *   H5SC_t                 *cache   – SCC instance holding global header LRU
 *   H5SC_dset_header_t     *dset_hdr  – Header to remove (must be linked)
 *
 * Returns
 *   SUCCEED on success; FAIL if the header is not a member or splice fails.
 *
 * Notes
 *   - Invariants are checked when H5SC_DO_SANITY_CHECKS is enabled.
 */

herr_t
H5SC__dset_lru_remove(H5SC_t *cache, H5SC_dset_header_t *dset_hdr)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *it = NULL, *pr = NULL;
    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

#if H5SC_DO_SANITY_CHECKS
    /* Must be a member */
    if (!(dset_hdr->next_dset_ptr || dset_hdr->prev_dset_ptr || cache->dset_lru_head_ptr == dset_hdr))
        HGOTO_ERROR(H5E_SCC, H5E_NOT_A_MEMBER, FAIL, "header not in global LRU");
#endif

    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);

    H5SC_DLL_REMOVE(dset_hdr, next_dset_ptr, prev_dset_ptr, cache->dset_lru_head_ptr,
                    cache->dset_lru_tail_ptr,
                    HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "header remove failed"));

    dset_hdr->last_op = H5SC_TAG_LRU_REMOVE;
    cache->dset_lru_len--;
    cache->SCC_quiescent_size -= dset_hdr->curr_dset_size;

#if H5SC_DO_SANITY_CHECKS
    H5SC_DLL_CHECK_LINKS(it, pr, cache->dset_lru_head_ptr, cache->dset_lru_tail_ptr, next_dset_ptr,
                         prev_dset_ptr,
                         HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL, "bad header LRU linkage"));
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__dset_lru_remove() */

/*-------------------------------------------------------------------------
 * Function: H5SC__dset_lru_promote
 *
 * Purpose:
 *   Function that promotes a dataset header to the most-recently-used
 *     (MRU) position in the global dataset LRU list maintained by SCC.
 *     If the header is already the current list head, no action is taken.
 *     If the header is linked elsewhere in the list, it is removed from
 *     its current position and then prepended to the front of the list.
 *
 *   This function is used to refresh dataset recency within the global
 *     LRU ordering after access or other operations that should move the
 *     associated dataset header to the head of the eviction queue.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state containing the global dataset LRU.
 *
 *   H5SC_dset_header_t *dset_hdr:
 *     Pointer to the dataset header to promote. The header must be a
 *     valid H5SC_dset_header_t instance.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
herr_t
H5SC__dset_lru_promote(H5SC_t *cache, H5SC_dset_header_t *dset_hdr)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset_hdr);
    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);

    /* If already MRU, nothing to do */
    if (cache->dset_lru_head_ptr == dset_hdr)
        HGOTO_DONE(SUCCEED);

    /* If linked, remove then prepend (accounting is handled by the ops) */
    if (dset_hdr->next_dset_ptr || dset_hdr->prev_dset_ptr || cache->dset_lru_head_ptr == dset_hdr) {
        if (H5SC__dset_lru_remove(cache, dset_hdr) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL, "failed to remove header from global LRU");
    }

    if (H5SC__dset_lru_prepend(cache, dset_hdr) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL, "failed to prepend header to global LRU");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__dset_lru_promote() */

/******************************************************************************
 *
 * Function: H5SC__chunk_lru_prepend
 *
 * Purpose:  Insert a chunk as MRU (head) of its dataset’s chunk LRU and update
 * counters by chunk->cached_chunk_size.
 *
 * Inputs:
 *   H5SC_dset_header_t *dset_hdr – Dataset header whose LRU is modified
 *   H5SC_chunk_t       *chunk    – Chunk to insert (must not already be linked)
 *
 * Returns
 *   SUCCEED on success; FAIL on already-linked, splice, or invariant error.
 */

herr_t
H5SC__chunk_lru_prepend(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk)
{
    H5SC_reclaim_contrib_t before;
    H5SC_reclaim_contrib_t after;
    herr_t                 ret_value = SUCCEED;
    H5SC_chunk_t          *it = NULL, *pr = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);
    assert(chunk->magic == H5SC_CHUNK_MAGIC);

#if H5SC_DO_SANITY_CHECKS
    if (chunk->next_ptr || chunk->prev_ptr || dset_hdr->lru_head_ptr == chunk)
        HGOTO_ERROR(H5E_SCC, H5E_ALREADY_LINKED, FAIL, "chunk already linked in per-dataset LRU");
#endif

    before = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    /* Splice first; if it fails, nothing changes. */
    H5SC_DLL_PREPEND(chunk, next_ptr, prev_ptr, dset_hdr->lru_head_ptr, dset_hdr->lru_tail_ptr,
                     HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "chunk prepend failed"));

    /* Embedded counters (authoritative for the per-dataset LRU). */
    dset_hdr->chunk_lru_len++;
    dset_hdr->curr_dset_size += chunk->cached_chunk_size;

#if H5SC_DO_SANITY_CHECKS
    assert(dset_hdr->curr_dset_size >= chunk->cached_chunk_size);
#endif

    /* Add the struct tags */
    dset_hdr->last_op = H5SC_TAG_LRU_TOUCH;
    chunk->last_op    = H5SC_TAG_LRU_PROMOTE;

    after = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    if (H5SC__apply_reclaim_transition(cache, dset_hdr, before, after) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                    "unable to account reclaimability after chunk LRU insertion");

#if H5SC_DO_SANITY_CHECKS
    H5SC_DLL_CHECK_LINKS(it, pr, dset_hdr->lru_head_ptr, dset_hdr->lru_tail_ptr, next_ptr, prev_ptr,
                         HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL, "bad chunk LRU linkage"));
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

/******************************************************************************
 *
 * Function:    H5SC__chunk_lru_remove
 *
 * Purpose:
 *   Remove a chunk from its dataset’s chunk LRU and decrement counters
 *   by chunk->cached_chunk_size.
 *
 * Inputs:
 *   H5SC_dset_header_t *dset_hdr – Dataset header whose LRU is modified
 * H5SC_chunk_t  *chunk           – Chunk to remove (must be linked)
 *
 * Returns
 *   SUCCEED on success; FAIL if the chunk is not a member or splice fails.
 */

herr_t
H5SC__chunk_lru_remove(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk)
{
    H5SC_reclaim_contrib_t before;
    H5SC_reclaim_contrib_t after;
    herr_t                 ret_value = SUCCEED;
    H5SC_chunk_t          *it = NULL, *pr = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);

    if (!(chunk->next_ptr || chunk->prev_ptr || dset_hdr->lru_head_ptr == chunk))
        HGOTO_ERROR(H5E_SCC, H5E_NOT_A_MEMBER, FAIL, "chunk not in per-dataset LRU");

    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);
    assert(chunk->magic == H5SC_CHUNK_MAGIC);

    before = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    /* Splice out first; if it fails, leave counters untouched. */
    H5SC_DLL_REMOVE(chunk, next_ptr, prev_ptr, dset_hdr->lru_head_ptr, dset_hdr->lru_tail_ptr,
                    HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "chunk remove failed"));

    chunk->last_op    = H5SC_TAG_LRU_REMOVE;
    dset_hdr->last_op = H5SC_TAG_LRU_REMOVE;

    /* Embedded counters */
    dset_hdr->chunk_lru_len--;

#if H5SC_DO_SANITY_CHECKS
    if (chunk->cached_chunk_size > dset_hdr->curr_dset_size)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "chunk bytes underflow");
#endif

    dset_hdr->curr_dset_size -= chunk->cached_chunk_size;

    after = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    if (H5SC__apply_reclaim_transition(cache, dset_hdr, before, after) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                    "unable to account reclaimability after chunk LRU removal");

#if H5SC_DO_SANITY_CHECKS
    H5SC_DLL_CHECK_LINKS(it, pr, dset_hdr->lru_head_ptr, dset_hdr->lru_tail_ptr, next_ptr, prev_ptr,
                         HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL, "bad chunk LRU linkage"));
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

/******************************************************************************
 *
 * Function:    H5SC__chunk_update_cached_size
 *
 * Purpose:
 *   Adjust counters to reflect a change in chunk->cached_chunk_size and commit
 *   the new size into the chunk object.
 *
 * Inputs:
 *   H5SC_t cache                - Pointer to the cache
 *   H5SC_dset_header_t dset_hdr – Dataset containing the target chunk
 *   H5SC_chunk_t *chunk         – Target chunk (must already exist in the cache)
 *   size_t new_size             – New cached size in bytes
 *
 * Returns
 *   SUCCEED on success; FAIL if size math would violate invariants.
 */
herr_t
H5SC__chunk_update_cached_size(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk,
                               size_t new_size)
{
    H5SC_reclaim_contrib_t before;
    H5SC_reclaim_contrib_t after;
    size_t                 old_size;
    size_t                 old_dset_size;
    bool                   linked;
    herr_t                 ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset_hdr);
    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);
    assert(chunk);
    assert(chunk->magic == H5SC_CHUNK_MAGIC);

    old_size      = chunk->cached_chunk_size;
    old_dset_size = dset_hdr->curr_dset_size;

    if (new_size == old_size)
        HGOTO_DONE(SUCCEED);

    linked = chunk->next_ptr || chunk->prev_ptr || dset_hdr->lru_head_ptr == chunk;

    before = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    if (linked) {
        if (new_size > old_size) {
            size_t delta = new_size - old_size;

            if (delta > SIZE_MAX - dset_hdr->curr_dset_size)
                HGOTO_ERROR(H5E_SCC, H5E_OVERFLOW, FAIL, "dataset cached-size accounting overflow");

            dset_hdr->curr_dset_size += delta;
        }
        else {
            size_t delta = old_size - new_size;

            if (delta > dset_hdr->curr_dset_size)
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "dataset cached-size accounting underflow");

            dset_hdr->curr_dset_size -= delta;
        }
    }

    chunk->cached_chunk_size = new_size;

    after = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    if (H5SC__apply_reclaim_transition(cache, dset_hdr, before, after) < 0) {
        /*
         * Restore both chunk and dataset accounting before reporting failure.
         */
        chunk->cached_chunk_size = old_size;

        if (linked)
            dset_hdr->curr_dset_size = old_dset_size;

        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                    "unable to update cached reclaimability after chunk resize");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5SC__chunk_update_cached_size() */

/* -------------------------------------------------------------
 * Debug helpers
 * -------------------------------------------------------------
 */
#if defined(H5SC_ENABLE_STAT_DUMPS) && (H5SC_ENABLE_STAT_DUMPS + 0)
static FILE *
cache_dbg_stream(FILE *s)
{
    return s ? s : stderr;
}

/******************************************************************************
 *
 * Function:    H5SC_dset_lru_dump
 *
 * Purpose:
 *   Debug helper to print the dataset-header LRU to a
 * stream.
 *
 * Inputs:
 *   const char *tag  – Label to prefix the line
 *   const H5SC_t *cache – SCC instance (provides head/tail
 * and counters) FILE *stream     – Output stream; if NULL,
 * stderr is used
 *
 * Returns
 *   void
 */

void
H5SC_dset_lru_dump(const char *tag, const H5SC_t *cache, FILE *stream)
{
    FILE *out = cache_dbg_stream(stream);
    if (!out)
        goto done;
    fprintf(out, "[HDR-LRU:%s] len=%llu quiescent_bytes=%llu :", tag ? tag : "",
            (unsigned long long)cache->dset_lru_len, (unsigned long long)cache->SCC_quiescent_size);
    for (const H5SC_dset_header_t *p = cache->dset_lru_head_ptr; p; p = p->next_dset_ptr)
        fprintf(out, " 0x%llx", (unsigned long long)p->dset_addr);
    fputc('\n', out);
done:;
}

/******************************************************************************
 *
 * Function:    H5SC_chunk_lru_dump
 *
 * Purpose:
 *   Debug helper to print a per-dataset chunk LRU to a
 * stream.
 *
 * Inputs:
 *   const char            *tag  – Label to prefix the line
 *   const H5SC_dset_header_t *dset_hdr – Header (provides
 * chunk head/tail & counters) FILE *stream               –
 * Output stream; if NULL, stderr is used
 *
 * Returns
 *   void
 */

void
H5SC_chunk_lru_dump(const char *tag, const H5SC_dset_header_t *dset_hdr, FILE *stream)
{
    FILE *out = cache_dbg_stream(stream);
    if (!out)
        goto done;
    fprintf(out, "[CHK-LRU:%s] len=%llu bytes=%llu :", tag ? tag : "",
            (unsigned long long)dset_hdr->chunk_lru_len, (unsigned long long)dset_hdr->curr_dset_size);
    for (const H5SC_chunk_t *p = dset_hdr->lru_head_ptr; p; p = p->next_ptr)
        fprintf(out, " %p", (const void *)p);
    fputc('\n', out);
done:;
}
#endif /* H5SC_ENABLE_STAT_DUMPS */

/*
 * UTHash Functions
 */

/******************************************************************************
 *
 * Function:    H5SC__hash_init
 *
 * Purpose:
 *   Initialize both hash-table head pointers on an H5SC_t state object. This
 *   routine performs no allocations and is exclusivly used in testing.
 *
 * Inputs:
 *   H5SC_t *cache – State object whose table heads will be initialized to NULL.
 *
 * Returns
 *   (void)
 */

void
H5SC__hash_init(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->chunk_hash_table_head_ptr = NULL;
    cache->dset_hash_table_head_ptr  = NULL;
}

/******************************************************************************
 *
 * Function:    H5SC__reset_hash_tables
 *
 * Purpose:
 *   Unlink every entry from both hash tables and set their head pointers to
 *   NULL. This routine does not free the unlinked entries; ownership and
 *   lifetime remain with the caller.
 *
 * Inputs:
 *   H5SC_t *cache – State object whose tables will be unlinked and reset
 *
 * Returns
 *   (void)
 */

void
H5SC__reset_hash_tables(H5SC_t *cache)
{
    H5SC_chunk_t       *chk, *chk_tmp;
    H5SC_dset_header_t *dset_hdr, *dset_hdr_tmp;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    H5SC_CHUNK_ITER(cache->chunk_hash_table_head_ptr, chk, chk_tmp)
    {
        H5SC_CHUNK_DEL(cache->chunk_hash_table_head_ptr, chk);
    }
    H5SC_DSET_ITER(cache->dset_hash_table_head_ptr, dset_hdr, dset_hdr_tmp)
    {
        H5SC_DSET_DEL(cache->dset_hash_table_head_ptr, dset_hdr);
    }

    cache->chunk_hash_table_head_ptr = NULL;
    cache->dset_hash_table_head_ptr  = NULL;
}

/******************************************************************************
 *
 * Function:    H5SC__ht_dset_insert
 *
 * Purpose:
 *   Insert a dataset header record into the dataset hash table, keyed by
 *   dset_hdr->dset_addr. If an entry with the same key already exists, this
 *   function is idempotent and performs no mutation of payload fields.
 *
 * Inputs:
 *   H5SC_t             *cache    – State object holding the dataset hash head
 *   H5SC_dset_header_t *dset_hdr – Dataset-header record to link
 *
 * Returns
 *   SUCCEED on success; FAIL on internal error (e.g., invalid arguments).
 */

herr_t
H5SC__ht_dset_insert(H5SC_t *cache, H5SC_dset_header_t *dset_hdr)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *found     = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(dset_hdr);

    if (dset_hdr->magic != H5SC_DSET_HDR_MAGIC)
        HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL, "Magic for dset header not properly set (ht insert)");

    H5SC_DSET_FIND(cache->dset_hash_table_head_ptr, &dset_hdr->dset_addr, found);
    if (!found) {
        H5SC_DSET_ADD(cache->dset_hash_table_head_ptr, dset_hdr);
        dset_hdr->last_op = H5SC_TAG_HT_INSERT;
    }
    else {
#ifdef DO_SANITY_CHECKS
        assert(found->magic == H5SC_DSET_HDR_MAGIC);
#endif
        found->last_op = H5SC_TAG_LOOKUP;
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

/******************************************************************************
 *
 * Function:    H5SC__ht_dset_find
 *
 * Purpose:
 *   Lookup a dataset header record by its haddr_t key and return the stored
 *   pointer. No allocations or payload mutations occur.
 *
 * Inputs:
 *   H5SC_t *cache  – State object holding the dataset hash head
 *   haddr_t addr   – Address key to search for
 *
 * Returns
 *   Pointer to the matching H5SC_dset_header_t on success; NULL if not found.
 */

H5SC_dset_header_t *
H5SC__ht_dset_find(H5SC_t *cache, haddr_t addr)
{
    H5SC_dset_header_t *found = NULL;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    H5SC_DSET_FIND(cache->dset_hash_table_head_ptr, &addr, found);

    if (found) {
        found->last_op = H5SC_TAG_LOOKUP;
    }

    return found;
}

/******************************************************************************
 *
 * Function:    H5SC__ht_dset_delete
 *
 * Purpose:
 *   Unlink (remove) a dataset-header record from the hash table by address.
 *   This does not free the unlinked node; ownership and lifetime remain with
 *   the caller.
 *
 * Inputs:
 *   H5SC_t *cache   – State object holding the dataset hash head
 *   haddr_t addr    – Address key to remove from the table
 *
 * Returns
 *   SUCCEED on success; FAIL if the key is not present.
 */

herr_t
H5SC__ht_dset_delete(H5SC_t *cache, haddr_t addr)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *found     = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    H5SC_DSET_FIND(cache->dset_hash_table_head_ptr, &addr, found);
    if (!found)
        HGOTO_ERROR(H5E_CACHE, H5E_NOTFOUND, FAIL, "dataset address not found");

    assert(found->magic == H5SC_DSET_HDR_MAGIC);
    H5SC_DSET_DEL(cache->dset_hash_table_head_ptr, found);
    found->last_op = H5SC_TAG_HT_DELETE;

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

/******************************************************************************
 *
 * Function:    H5SC__ht_chunk_insert
 *
 * Purpose:
 *   Insert a chunk record into the chunk hash table, keyed by node->data_key
 *   (128-bit binary key). If an entry with the same key already exists, this
 *   function is idempotent and performs no mutation of payload fields.
 *
 * Inputs:
 *   H5SC_t        *cache   – State object holding the chunk hash head
 *   H5SC_chunk_t  *node    – Chunk record to link (payload must be initialized)
 *
 * Returns
 *   SUCCEED on success; FAIL on internal error (e.g., invalid arguments).
 */

herr_t
H5SC__ht_chunk_insert(H5SC_t *cache, H5SC_chunk_t *chk)
{
    herr_t        ret_value = SUCCEED;
    H5SC_chunk_t *found     = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(chk);

    H5SC_CHUNK_FIND(cache->chunk_hash_table_head_ptr, &chk->data_key, found);
    if (!found) {
        assert(chk->magic == H5SC_CHUNK_MAGIC);
        H5SC_CHUNK_ADD(cache->chunk_hash_table_head_ptr, chk);
        H5SC__stats_record_insert(cache);
        chk->last_op = H5SC_TAG_HT_INSERT;
    }
    else {
        assert(found->magic == H5SC_CHUNK_MAGIC);
        found->last_op = H5SC_TAG_HT_INSERT;
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL, "attempted to insert existing chunk");
    }

    /* Update struct tag to reflect it has been added to the hash table */

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

/******************************************************************************
 *
 * Function:    H5SC__ht_chunk_find
 *
 * Purpose:
 *   Lookup a chunk record by its 128-bit key and return the stored pointer.
 *   No allocations or payload mutations occur.
 *
 * Inputs:
 *   H5SC_t                *cache – State object holding the chunk hash head
 *   const H5SC_chunk_key_t *k – Address of the 128-bit key to search for
 *
 * Returns
 *   Pointer to the matching H5SC_chunk_t on success; NULL if not found.
 */

H5SC_chunk_t *
H5SC__ht_chunk_find(H5SC_t *cache, const H5SC_chunk_key_t *chk_key)
{
    H5SC_chunk_t *found = NULL;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(chk_key);

    H5SC_CHUNK_FIND(cache->chunk_hash_table_head_ptr, chk_key, found);
    if (found) {
        found->last_op = H5SC_TAG_LOOKUP;
        H5SC__stats_record_lookup(cache, true);
    }
    else {
        H5SC__stats_record_lookup(cache, false);
    }

    return found;
} /* end H5SC__ht_chunk_find() */

/******************************************************************************
 *
 * Function:    H5SC__ht_chunk_delete
 *
 * Purpose:
 *   Unlink (remove) a chunk record from the hash table by key. This does not
 *   free the unlinked node; ownership and lifetime remain with the caller.
 *
 * Inputs:
 *   H5SC_t                *cache – State object holding the chunk hash head
 *   const H5SC_chunk_key_t *k    – Address of the 128-bit key to remove
 *
 * Returns
 *   SUCCEED on success; FAIL if the key is not present.
 */

herr_t
H5SC__ht_chunk_delete(H5SC_t *cache, const H5SC_chunk_key_t *chk_key)
{
    herr_t        ret_value = SUCCEED;
    H5SC_chunk_t *found     = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);
    assert(chk_key);

    H5SC_CHUNK_FIND(cache->chunk_hash_table_head_ptr, chk_key, found);
    if (!found)
        HGOTO_ERROR(H5E_SCC, H5E_HT_NOTFOUND, FAIL, "chunk key not found");

    assert(found->magic == H5SC_CHUNK_MAGIC);
    H5SC_CHUNK_DEL(cache->chunk_hash_table_head_ptr, found);
    H5SC__stats_record_delete(cache);
    found->last_op = H5SC_TAG_HT_DELETE;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__ht_chunk_delete() */

/*-------------------------------------------------------------------------
 * Function:    H5SC_validate_config()
 *
 * Purpose:     Run a sanity check on the contents of the supplied
 *              instance of H5SC__cache_config_t.
 *
 *              Do nothing and return SUCCEED if no errors are detected,
 *              and flag an error and return FAIL otherwise.
 *
 *              At present, this function just returns true -- fill it
 *              out as appropriate.
 *
 * Return:      Non-negative on success/Negative on failure
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_validate_config(const H5SC__cache_config_t *config_ptr)
{
    herr_t ret_value = SUCCEED; /* Return value */

    FUNC_ENTER_NOAPI(FAIL)

    /* Check args */
    if (config_ptr == NULL)
        HGOTO_ERROR(H5E_CACHE, H5E_BADVALUE, FAIL, "NULL config_ptr on entry");
    if (config_ptr->version != H5SC__CURR_SCC_VERSION)
        HGOTO_ERROR(H5E_CACHE, H5E_BADVALUE, FAIL, "Unknown config version");

    /* add validation code here */
    if (config_ptr->max_q_size == (size_t)0) {
        HGOTO_ERROR(H5E_SCC, H5E_INVALIDLIMIT, FAIL, "Quiescent limit must be non-zero");
    }

    if (config_ptr->max_a_size == (size_t)0) {
        HGOTO_ERROR(H5E_SCC, H5E_INVALIDLIMIT, FAIL, "Active limit must be non-zero");
    }

    if (config_ptr->max_q_size > config_ptr->max_a_size) {
        HGOTO_ERROR(H5E_SCC, H5E_INVALIDLIMIT, FAIL, "Active limit must be greater than the quiescent limit");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5SC_validate_config() */

static bool
H5SC__space_exceeded(size_t current, size_t additional, size_t limit)
{
    if (additional > limit)
        return true;

    return current > (limit - additional);
}

static void
H5SC__saturating_add_size(size_t *total, size_t amount)
{
    assert(total);

    if (amount > SIZE_MAX - *total)
        *total = SIZE_MAX;
    else
        *total += amount;
}

/*-------------------------------------------------------------------------
 * Function: H5SC__cached_active_reclaimable
 *
 * Purpose:
 *   Return the cache-wide number of resident bytes reclaimable under the
 *   active-pressure policy. The result combines cached clean-eviction bytes
 *   and cached dirty flush-and-evict bytes, saturating at SIZE_MAX.
 *
 *   Dataset-local min_dset_size targets are intentionally not applied.
 *-------------------------------------------------------------------------
 */
static inline size_t
H5SC__cached_active_reclaimable(const H5SC_t *cache)
{
    size_t reclaimable;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    reclaimable = cache->reclaimable_clean_bytes;

    if (cache->reclaimable_dirty_bytes > SIZE_MAX - reclaimable)
        return SIZE_MAX;

    return reclaimable + cache->reclaimable_dirty_bytes;
}

/*-------------------------------------------------------------------------
 * Function: H5SC__effective_budget
 *
 * Purpose:
 *   Return the additional resident bytes that may be admitted under the
 *   active-pressure policy. The budget consists of unused active-limit
 *   headroom plus resident bytes currently reclaimable by clean eviction or
 *   dirty flush-and-evict. Arithmetic saturates at SIZE_MAX.
 *
 *   Pinned chunks are excluded through the cached reclaimability counters.
 *   Dataset-local min_dset_size targets are ignored because they apply only
 *   to quiescent retention.
 *-------------------------------------------------------------------------
 */
static inline size_t
H5SC__effective_budget(const H5SC_t *cache)
{
    size_t budget      = 0;
    size_t reclaimable = 0;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    if (cache->SCC_quiescent_size < cache->SCC_active_limit)
        budget = cache->SCC_active_limit - cache->SCC_quiescent_size;

    reclaimable = H5SC__cached_active_reclaimable(cache);

    if (reclaimable > SIZE_MAX - budget)
        return SIZE_MAX;

    return budget + reclaimable;
}

/*-------------------------------------------------------------------------
 * Function: H5SC__chunk_pin
 *
 * Purpose:
 *   Acquire one request pin and remove the chunk's contribution from cached
 *   active-pressure reclaimability when the pin count changes from zero to
 *   one. Additional nested pins do not change reclaimability.
 *-------------------------------------------------------------------------
 */
static inline herr_t
H5SC__chunk_pin(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk)
{
    H5SC_reclaim_contrib_t before;
    H5SC_reclaim_contrib_t after;

    assert(cache);
    assert(dset_hdr);
    assert(chunk);

    if (chunk->chunk_counter == SIZE_MAX)
        return FAIL;

    before = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    chunk->chunk_counter++;

    after = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    if (H5SC__apply_reclaim_transition(cache, dset_hdr, before, after) < 0) {
        chunk->chunk_counter--;
        return FAIL;
    }

    return SUCCEED;
} /* end H5SC__chunk_pin() */

/*-------------------------------------------------------------------------
 * Function: H5SC__chunk_unpin
 *
 * Purpose:
 *   Release one request pin and add the chunk's current clean or dirty
 *   resident-byte contribution when the final pin is released.
 *-------------------------------------------------------------------------
 */
static inline herr_t
H5SC__chunk_unpin(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk)
{
    H5SC_reclaim_contrib_t before;
    H5SC_reclaim_contrib_t after;

    assert(cache);
    assert(dset_hdr);
    assert(chunk);

    if (chunk->chunk_counter == 0)
        return FAIL;

    before = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    chunk->chunk_counter--;

    after = H5SC__chunk_reclaim_contribution(dset_hdr, chunk);

    if (H5SC__apply_reclaim_transition(cache, dset_hdr, before, after) < 0) {
        chunk->chunk_counter++;
        return FAIL;
    }

    return SUCCEED;
} /* end H5SC__chunk_unpin() */

#if H5SC_DO_SANITY_CHECKS
/* Test-only entry points for exercising centralized chunk state transitions. */
herr_t
H5SC__test_chunk_pin(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk)
{
    return H5SC__chunk_pin(cache, dset_hdr, chunk);
}

herr_t
H5SC__test_chunk_unpin(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk)
{
    return H5SC__chunk_unpin(cache, dset_hdr, chunk);
}

herr_t
H5SC__test_chunk_set_dirty(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk, bool dirty)
{
    return H5SC__chunk_set_dirty(cache, dset_hdr, chunk, dirty);
}

herr_t
H5SC__test_ensure_space(H5SC_t *cache, size_t bytes_needed)
{
    return H5SC__ensure_space(cache, bytes_needed, false);
}

herr_t
H5SC__test_ensure_oversized_single_chunk_space(H5SC_t *cache, size_t bytes_needed)
{
    return H5SC__ensure_space(cache, bytes_needed, true);
}

herr_t
H5SC__test_trim_to_quiescent_limit(H5SC_t *cache)
{
    return H5SC__trim_to_quiescent_limit(cache);
}

herr_t
H5SC__test_account_chunk_link_change(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, size_t old_dset_size)
{
    return H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size);
}
#endif

/*
 *
 ******************************************************************************
 *                      H5SC_stats_t Related Functions
 ******************************************************************************
 *
 */
/******************************************************************************
 *
 * Function:    H5SC__stats_reset
 *
 * Purpose:
 *   Reset all SCC statistics counters for a cache instance to 0.
 *
 * Inputs:
 *   H5SC_t *cache
 *       SCC instance whose statistics are to be reset.
 *
 * Returns
 *   No return value.
 *
 * Notes
 *   - This routine clears the entire H5SC_stats_t structure stored within
 *     the supplied cache instance.
 *   - Intended to be called during SCC creation, destruction, or explicit
 *     reinitialization.
 */

void
H5SC__stats_reset(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    memset(&cache->stats, 0, sizeof(cache->stats));
} /* end H5SC__stats_reset() */

/******************************************************************************
 *
 * Function:    H5SC__stats_record_lookup
 *
 * Purpose
 *   Record a chunk lookup in the SCC and increment either the hit or miss
 * counter depending on whether the lookup resolved to a resident
 * non-placeholder chunk.
 *
 * Inputs:
 *   H5SC_t  *cache
 *       SCC instance whose statistics are to be updated.
 *   hbool_t  hit
 *       Boolean indicating whether the lookup was a cache hit.
 *
 * Returns
 *   No return value.
 *
 * Notes
 *   - A hit indicates that the requested chunk is a resident non-placeholder
 *     chunk already present within the cache.
 *   - A miss indicates that the requested chunk was not found in that form.
 */

void
H5SC__stats_record_lookup(H5SC_t *cache, hbool_t hit)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_lookups++;

    if (hit)
        cache->stats.scc_hits++;
    else
        cache->stats.scc_misses++;
} /* end H5SC__stats_record_lookup() */

/******************************************************************************
 *
 * Function:    H5SC__stats_record_insert
 *
 * Purpose
 *   Record a chunk insert operation in the SCC statistics for a cache
 * instance.
 *
 * Inputs:
 *   H5SC_t *cache
 *       SCC instance whose statistics are to be updated.
 *
 * Returns
 *   No return value.
 *
 * Notes
 *   - This routine should be called only when a chunk is actually inserted
 *     into the SCC, not when an existing cached chunk is merely updated.
 */

void
H5SC__stats_record_insert(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_inserts++;
} /* end H5SC__stats_record_insert() */

/******************************************************************************
 *
 * Function:    H5SC__stats_record_delete
 *
 * Purpose
 *   Record a chunk remove operation in the SCC statistics for a cache
 * instance.
 *
 * Inputs:
 *   H5SC_t *cache
 *       SCC instance whose statistics are to be updated.
 *
 * Returns
 *   No return value.
 *
 * Notes
 *   - This routine should be used for explicit removals from the SCC that
 *     are logically distinct from eviction.
 */

void
H5SC__stats_record_delete(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_removes++;
} /* end H5SC__stats_record_delete() */

/******************************************************************************
 *
 * Function:    H5SC__stats_record_reads
 *
 * Purpose
 *   Record a batch read operation in the SCC statistics for a cache
 * instance.
 *
 * Inputs:
 *   H5SC_t *cache
 *       SCC instance whose statistics are to be updated.
 *
 * Returns
 *   No return value.
 *
 */

void
H5SC__stats_record_batch_read(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_reads_batch_calls_count++;
} /* end H5SC__stats_record_batch_read() */

/******************************************************************************
 *
 * Function:    H5SC__stats_record_writes
 *
 * Purpose
 *   Record a batch write operation in the SCC statistics for a cache
 * instance.
 *
 * Inputs:
 *   H5SC_t *cache
 *       SCC instance whose statistics are to be updated.
 *
 * Returns
 *   No return value.
 *
 */

void
H5SC__stats_record_batch_write(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_writes_batch_calls_count++;
} /* end H5SC__stats_record_batch_write() */

/*-------------------------------------------------------------------------
 * Function: H5SC__stats_record_batch_read_len
 *
 * Purpose:
 *   Record the size of the most recent read batch and update the maximum
 *   observed batch length if applicable.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     SCC instance whose statistics are to be updated.
 *
 *   size_t batch_len:
 *     Number of elements processed in the batch.
 *
 * Return:
 *   No return value.
 *-------------------------------------------------------------------------
 */
void
H5SC__stats_record_batch_read_len(H5SC_t *cache, size_t batch_len)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_reads_last_batch_len = batch_len;

    if (batch_len > cache->stats.scc_reads_max_batch_len)
        cache->stats.scc_reads_max_batch_len = batch_len;
}

/*-------------------------------------------------------------------------
 * Function: H5SC__stats_record_batch_write_len
 *
 * Purpose:
 *   Record the size of the most recent write batch and update the maximum
 *   observed batch length if applicable.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     SCC instance whose statistics are to be updated.
 *
 *   size_t batch_len:
 *     Number of elements processed in the batch.
 *
 * Return:
 *   No return value.
 *-------------------------------------------------------------------------
 */
void
H5SC__stats_record_batch_write_len(H5SC_t *cache, size_t batch_len)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_writes_last_batch_len = batch_len;

    if (batch_len > cache->stats.scc_writes_max_batch_len)
        cache->stats.scc_writes_max_batch_len = batch_len;
}

/******************************************************************************
 *
 * Function:    H5SC__stats_record_chunk_flush
 *
 * Purpose
 *   Record a chunk flush operation in the SCC statistics for a cache instance.
 *
 * Inputs:
 *   H5SC_t *cache
 *       SCC instance whose statistics are to be updated.
 *
 * Returns
 *   No return value.
 *
 * Notes
 *   - This routine should be called at the point where SCC-managed chunks are
 *     actually flushed, rather than at a higher-level request boundary, to
 *     avoid overcounting.
 */
void
H5SC__stats_record_chunk_flush(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_chunk_flush_count++;
} /* end H5SC__stats_record_chunk_flush() */

/******************************************************************************
 *
 * Function:    H5SC__stats_record_dset_flush
 *
 * Purpose
 *   Record a flush operation in the SCC statistics for a cache instance.
 *
 * Inputs:
 *   H5SC_t *cache
 *       SCC instance whose statistics are to be updated.
 *
 * Returns
 *   No return value.
 *
 * Notes
 *   - This routine should be called at the point where SCC-managed data is
 *     actually flushed, rather than at a higher-level request boundary, to
 *     avoid overcounting.
 */
void
H5SC__stats_record_dset_flush(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_dataset_flush_count++;
} /* end H5SC__stats_record_dset_flush() */

/******************************************************************************
 *
 * Function:    H5SC__stats_record_eviction
 *
 * Purpose
 *   Record an eviction operation in the SCC statistics for a cache instance.
 *
 * Inputs:
 *   H5SC_t *cache
 *       SCC instance whose statistics are to be updated.
 *
 * Returns
 *   No return value.
 *
 * Notes
 *   - This routine should be called when a chunk or other SCC-managed object
 *     is evicted from the cache due to cache management policy.
 *   - Explicit removals that are not policy-driven evictions should be
 *     tracked separately using H5SC__stats_record_delete().
 */
void
H5SC__stats_record_eviction(H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.scc_evictions++;
} /* end H5SC__stats_record_eviction() */

/*-------------------------------------------------------------------------
 * Function: H5SC__estimate_resident_size
 *
 * Purpose:
 *   Estimate the final resident allocation required to materialize a
 *   nonresident SCC chunk for admission and batch-budget calculations.
 *
 *   The result is an admission heuristic, not a formal allocation upper
 *   bound. Dataset-local history is preferred after a successful
 *   materialization because sparse resident allocations may be substantially
 *   smaller than the dense logical chunk size. The dataset's dense logical
 *   chunk size is used as the cold-dataset fallback when no history is
 *   available.
 *
 *   The current implementation uses, in priority order:
 *
 *     1. the maximum of the dataset-local exponential moving average and
 *        decaying high-water resident-size observations;
 *
 *     2. the dataset's dense logical chunk size when no resident-size
 *        observations are available.
 *
 *   H5SC_io_sel_chunk_t::lookup_size_hint is not currently used as a resident
 *   estimate. For the structured-chunk layout, that field contains the
 *   allocation size of the encoded on-disk block used by the raw read/decode
 *   path. It does not describe the complete decoded resident allocation.
 *
 *   lookup_defined_values_size and lookup_defined_values_size_hint are byte
 *   sizes associated with encoded defined-values selection metadata. Neither
 *   field is a count of defined elements, and neither is currently used as
 *   nvalues in a sparse resident-size model.
 *
 *   Decoded sparse-selection metadata may cause the final resident allocation
 *   to exceed the dense logical data size, especially for small or
 *   higher-rank chunks. After successful materialization, callers must
 *   reconcile the estimate with the layout-reported resident allocation
 *   through H5SC__chunk_update_cached_size() and update dataset-local history
 *   through H5SC__update_resident_estimate_history().
 *
 *   The sparse-model and lookup estimate sources remain reserved for future
 *   use. They must not be selected until their inputs have semantics that
 *   describe, or can reliably predict, the complete decoded resident
 *   allocation.
 *
 *   WRITE_BYTES is accepted for a possible future conservative write-growth
 *   allowance. Current callers pass zero, so no write-growth term is included
 *   in the estimate.
 *
 * Inputs:
 *   const H5SC_dset_header_t *dset_hdr:
 *     Dataset-local SCC header containing adaptive resident-size history.
 *
 *   const H5D_t *dset:
 *     Dataset whose dense logical chunk size supplies the cold fallback.
 *
 *   const H5SC_io_sel_chunk_t *sel:
 *     Selected-chunk request state. Lookup-related fields are available for
 *     future estimator development but are not currently used.
 *
 *   bool is_write:
 *     True for write admission and false for read admission.
 *
 *   size_t write_bytes:
 *     Optional future write-growth allowance. Current callers pass zero.
 *
 * Outputs:
 *   H5SC_size_est_source_t *source:
 *     Receives H5SC_SIZE_EST_DATASET_HISTORY or
 *     H5SC_SIZE_EST_DENSE_FALLBACK for the current implementation.
 *
 *   size_t *estimate:
 *     Receives the predicted resident allocation in bytes.
 *
 * Return:
 *   SUCCEED when a nonzero estimate is produced;
 *   FAIL when no usable estimate can be produced.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__estimate_resident_size(const H5SC_dset_header_t *dset_hdr, const H5D_t *dset,
                             const H5SC_io_sel_chunk_t *sel, bool is_write, size_t write_bytes,
                             H5SC_size_est_source_t *source, size_t *estimate)
{
    size_t result  = 0;
    size_t history = 0;
    size_t dense;

    assert(dset_hdr);
    assert(dset);
    assert(sel);
    assert(source);
    assert(estimate);

    *source   = H5SC_SIZE_EST_NONE;
    *estimate = 0;

    dense = (size_t)dset->shared->layout.u.struct_chunk.size;

    /*
     * lookup_size_hint currently describes the encoded on-disk block
     * allocation for the structured-chunk layout. It is retained for raw
     * read-buffer allocation but is not used here as a decoded resident-size
     * estimate.
     */
    if (dset_hdr->resident_estimate_samples > 0) {
        history = MAX(dset_hdr->resident_estimate_ema, dset_hdr->resident_estimate_high);

        if (history > result) {
            result  = history;
            *source = H5SC_SIZE_EST_DATASET_HISTORY;
        }
    }

    if (result == 0) {
        result  = dense;
        *source = H5SC_SIZE_EST_DENSE_FALLBACK;
    }

    /*
     * Reserved for future conservative write-growth estimation. Current
     * callers pass write_bytes == 0.
     */
    if (is_write && write_bytes > 0)
        H5SC__saturating_add_size(&result, write_bytes);

    if (result == 0)
        return FAIL;

    *estimate = result;
    return SUCCEED;
} /* end H5SC__estimate_resident_size() */

static void
H5SC__report_oversized_admission(H5SC_t *cache, size_t estimated_bytes)
{
    size_t excess;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    excess = estimated_bytes > cache->SCC_active_limit ? estimated_bytes - cache->SCC_active_limit : 0;

    cache->stats.scc_oversized_admission_count++;

    if (estimated_bytes > cache->stats.scc_oversized_admission_max_bytes)
        cache->stats.scc_oversized_admission_max_bytes = estimated_bytes;

    if (excess > cache->stats.scc_oversized_admission_max_excess)
        cache->stats.scc_oversized_admission_max_excess = excess;

    fprintf(stderr,
            "HDF5 SCC warning: temporarily exceeded the configured active "
            "cache limit to process an indivisible chunk requiring "
            "approximately %zu byte(s(); the configured active limit is "
            "%zu byte(s). Consider increasing the active limit for datasets "
            "with large chunks.\n",
            estimated_bytes, cache->SCC_active_limit);
} /* end H5SC__report_oversized_admission() */

#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
/*-------------------------------------------------------------------------
 * Function: H5SC__stats_record_size_estimate
 *
 * Purpose:
 *   Record one completed comparison between a pending admission estimate and
 *   the successfully observed final resident allocation.
 *
 *   The function updates aggregate estimated and actual byte totals,
 *   over/under counts and byte deltas, maximum observed errors, and the count
 *   associated with SOURCE. Exact comparisons increase only the total and
 *   source counts.
 *-------------------------------------------------------------------------
 */
static void
H5SC__stats_record_size_estimate(H5SC_t *cache, H5SC_size_est_source_t source, size_t estimate, size_t actual)
{
    size_t delta;

    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    cache->stats.size_estimate_count++;

    H5SC__saturating_add_size(&cache->stats.size_estimated_total, estimate);
    H5SC__saturating_add_size(&cache->stats.size_actual_total, actual);

    switch (source) {
        case H5SC_SIZE_EST_LOOKUP:
            cache->stats.size_estimate_lookup_count++;
            break;

        case H5SC_SIZE_EST_SPARSE_MODEL:
            cache->stats.size_estimate_sparse_model_count++;
            break;

        case H5SC_SIZE_EST_DATASET_HISTORY:
            cache->stats.size_estimate_dataset_history_count++;
            break;

        case H5SC_SIZE_EST_DENSE_FALLBACK:
            cache->stats.size_estimate_dense_fallback_count++;
            break;

        case H5SC_SIZE_EST_NONE:
        default:
            break;
    }

    if (estimate > actual) {
        delta = estimate - actual;

        cache->stats.size_estimate_over_count++;
        H5SC__saturating_add_size(&cache->stats.size_overestimated_bytes, delta);

        if (delta > cache->stats.size_estimate_max_over)
            cache->stats.size_estimate_max_over = delta;
    }
    else if (actual > estimate) {
        delta = actual - estimate;

        cache->stats.size_estimate_under_count++;
        H5SC__saturating_add_size(&cache->stats.size_underestimated_bytes, delta);

        if (delta > cache->stats.size_estimate_max_under)
            cache->stats.size_estimate_max_under = delta;
    }
} /* end H5SC__stats_record_size_estimate() */
#endif

#if defined(H5SC_ENABLE_STAT_DUMPS) && (H5SC_ENABLE_STAT_DUMPS + 0)
/******************************************************************************
 *
 * Function:    H5SC__stats_get_hit_rate
 *
 * Purpose
 *   Return the SCC cache hit rate for a cache instance as a floating-point
 * ratio in the range [0.0, 1.0].
 *
 * Inputs:
 *   const H5SC_t *cache
 *       SCC instance whose statistics are to be queried.
 *
 * Returns
 *   The cache hit rate as a double. Returns 0.0 if no lookups have been
 *   recorded.
 *
 * Notes
 *   - A hit indicates that the requested chunk lookup resolved to a resident
 *     non-placeholder chunk already present within the cache.
 */

double
H5SC__stats_get_hit_rate(const H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    if (0 == cache->stats.scc_lookups)
        return 0.0;

    return ((double)cache->stats.scc_hits / (double)cache->stats.scc_lookups);
} /* end H5SC__stats_get_hit_rate() */

/******************************************************************************
 *
 * Function:    H5SC__stats_get_miss_rate
 *
 * Purpose
 *   Return the SCC cache miss rate for a cache instance as a floating-point
 * ratio in the range [0.0, 1.0].
 *
 * Inputs:
 *   const H5SC_t *cache
 *       SCC instance whose statistics are to be queried.
 *
 * Returns
 *   The cache miss rate as a double. Returns 0.0 if no lookups have been
 *   recorded.
 *
 * Notes
 *   - A miss indicates that the requested chunk lookup did not resolve to a
 *     resident non-placeholder chunk already present within the cache.
 */

double
H5SC__stats_get_miss_rate(const H5SC_t *cache)
{
    assert(cache);
    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    if (0 == cache->stats.scc_lookups)
        return 0.0;

    return ((double)cache->stats.scc_misses / (double)cache->stats.scc_lookups);
} /* end H5SC__stats_get_miss_rate() */

/******************************************************************************
 *
 * Function:    H5SC__stats_dump
 *
 * Purpose
 *   Print SCC statistics for a cache instance to the supplied output stream.
 *
 * Inputs:
 *   const H5SC_t *cache
 *       SCC instance whose statistics are to be printed.
 *   FILE        *stream
 *       Output stream receiving the formatted statistics.
 *
 * Returns
 *   SUCCEED on success; FAIL if the cache or stream pointer is invalid.
 *
 * Notes
 *   - The printed statistics include lookups, hits, misses, hit rate, miss
 *     rate, inserts, removes, flushes, and evictions.
 *   - Intended primarily for debugging and test instrumentation.
 */
static herr_t
H5SC__stats_dump(const H5SC_t *cache)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    if (!cache)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "invalid cache pointer");

    assert(cache->SCC_magic == H5SC_MAIN_MAGIC);

    fprintf(stdout, "H5SC cache statistics:\n");
    fprintf(stdout, "    lookups:   %" PRIuHSIZE "\n", cache->stats.scc_lookups);
    fprintf(stdout, "    hits:      %" PRIuHSIZE "\n", cache->stats.scc_hits);
    fprintf(stdout, "    misses:    %" PRIuHSIZE "\n", cache->stats.scc_misses);
    fprintf(stdout, "    hit rate:  %.2f%%\n", 100.0 * H5SC__stats_get_hit_rate(cache));
    fprintf(stdout, "    miss rate: %.2f%%\n", 100.0 * H5SC__stats_get_miss_rate(cache));
    fprintf(stdout, "    inserts:   %" PRIuHSIZE "\n", cache->stats.scc_inserts);
    fprintf(stdout, "    removes:   %" PRIuHSIZE "\n\n", cache->stats.scc_removes);
    fprintf(stdout, "    chunk flushes:    %" PRIuHSIZE "\n", cache->stats.scc_chunk_flush_count);
    fprintf(stdout, "    dataset flushes:  %" PRIuHSIZE "\n", cache->stats.scc_dataset_flush_count);
    fprintf(stdout, "    chunk evictions: %" PRIuHSIZE "\n\n", cache->stats.scc_evictions);
    fprintf(stdout, "    batched reads count:     %" PRIuHSIZE "\n",
            cache->stats.scc_reads_batch_calls_count);
    fprintf(stdout, "    batched reads last len:  %" PRIuHSIZE "\n", cache->stats.scc_reads_last_batch_len);
    fprintf(stdout, "    batched reads max len:   %" PRIuHSIZE "\n", cache->stats.scc_reads_max_batch_len);
    fprintf(stdout, "    batched writes count:    %" PRIuHSIZE "\n",
            cache->stats.scc_writes_batch_calls_count);
    fprintf(stdout, "    batched writes last len: %" PRIuHSIZE "\n", cache->stats.scc_writes_last_batch_len);
    fprintf(stdout, "    batched writes max len:  %" PRIuHSIZE "\n", cache->stats.scc_writes_max_batch_len);
    fprintf(stdout, "\n  oversized single-chunk admissions:\n");
    fprintf(stdout, "    admissions:              %" PRIu64 "\n", cache->stats.scc_oversized_admission_count);
    fprintf(stdout, "    maximum estimated bytes: %zu\n", cache->stats.scc_oversized_admission_max_bytes);
    fprintf(stdout, "    maximum excess bytes:    %zu\n", cache->stats.scc_oversized_admission_max_excess);

#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
    fprintf(stdout, "\n  resident-size estimates:\n");
    fprintf(stdout, "    comparisons:             %" PRIu64 "\n", cache->stats.size_estimate_count);
    fprintf(stdout, "    underestimates:          %" PRIu64 "\n", cache->stats.size_estimate_under_count);
    fprintf(stdout, "    overestimates:           %" PRIu64 "\n", cache->stats.size_estimate_over_count);
    fprintf(stdout, "    estimated bytes total:   %zu\n", cache->stats.size_estimated_total);
    fprintf(stdout, "    actual bytes total:      %zu\n", cache->stats.size_actual_total);
    fprintf(stdout, "    excess estimate bytes:   %zu\n", cache->stats.size_overestimated_bytes);
    fprintf(stdout, "    missed estimate bytes:   %zu\n", cache->stats.size_underestimated_bytes);
    fprintf(stdout, "    maximum overestimate:    %zu\n", cache->stats.size_estimate_max_over);
    fprintf(stdout, "    maximum underestimate:   %zu\n", cache->stats.size_estimate_max_under);

    fprintf(stdout, "    lookup-hint source:      %" PRIu64 "\n", cache->stats.size_estimate_lookup_count);
    fprintf(stdout, "    sparse-model source:     %" PRIu64 "\n",
            cache->stats.size_estimate_sparse_model_count);
    fprintf(stdout, "    dataset-history source:  %" PRIu64 "\n",
            cache->stats.size_estimate_dataset_history_count);
    fprintf(stdout, "    dense-fallback source:   %" PRIu64 "\n",
            cache->stats.size_estimate_dense_fallback_count);

    fprintf(stdout, "\n  resident write growth:\n");
    fprintf(stdout, "    observations:            %" PRIu64 "\n", cache->stats.write_growth_count);
    fprintf(stdout, "    total growth bytes:      %zu\n", cache->stats.write_growth_total);
    fprintf(stdout, "    maximum growth bytes:    %zu\n", cache->stats.write_growth_max);
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__stats_dump() */
#endif

/******************************************************************************
 *
 * Function:    H5SC__get_cache_from_file_id
 *
 * Purpose:
 *   Retrieve the SCC cache instance (H5SC_t) associated with the file
 *   identified by the provided file ID. This function is intended for
 *   internal use and testing, allowing access to SCC state from a public
 *   file ID by unwrapping the file VOL object through the H5VL module.
 *
 * Inputs:
 *   hid_t file_id
 *       Identifier of the file whose SCC cache is to be retrieved.
 *
 *   H5SC_t **cache
 *       Output pointer that will be set to the SCC instance associated with
 *       the file on success.
 *
 * Returns:
 *   SUCCEED on success;
 *   FAIL on failure.
 *
 ******************************************************************************/
herr_t
H5SC__get_cache_from_file_id(hid_t file_id, H5SC_t **cache)
{
    herr_t         ret_value    = SUCCEED;
    H5VL_object_t *file_vol_obj = NULL;
    void          *tmp_cache    = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);

    *cache = NULL;

    if (H5I_FILE != H5I_get_type(file_id))
        HGOTO_ERROR(H5E_SCC, H5E_BADTYPE, FAIL, "ID is not a file ID");

    if (NULL == (file_vol_obj = (H5VL_object_t *)H5I_object_verify(file_id, H5I_FILE)))
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTGET, FAIL, "failed to verify file ID");

    if (H5VL__get_file_shared_cache(file_vol_obj, &tmp_cache) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTGET, FAIL, "failed to get pointer to H5SC_t structure");

    *cache = (H5SC_t *)tmp_cache;
    assert((*cache)->SCC_magic == H5SC_MAIN_MAGIC);

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__get_cache_from_file_id() */