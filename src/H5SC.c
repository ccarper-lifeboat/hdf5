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

/********************/
/* Local Prototypes */
/********************/

/* I/O initialization related functions */
static inline void H5SC__io_sel_chunk_init(H5SC_io_sel_chunk_t *chunk);
static herr_t      H5SC__io_info_init(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info);
static herr_t      H5SC__erase_io_info_init(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space);
static herr_t      H5SC__io_info_reset(H5SC_io_info_t *sc_io_info);
static herr_t      H5SC__io_info_term(H5SC_io_info_t *sc_io_info);

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

static herr_t H5SC__compute_logical_chunk_index(unsigned ndims, const hsize_t *dset_dims, hsize_t *chunk_dims,
                                                const hsize_t *elem_coord, hsize_t *log_chk_idx);

static inline void H5SC__set_interleave_bit(H5SC_chunk_key_t *chunk_key, unsigned bit_position);

static herr_t H5SC__compute_chunk_key(haddr_t *dset_object_header_addr /*in*/, hsize_t *log_chk_coord /*in*/,
                                      H5SC_chunk_key_t *chunk_key /*in,out*/);

/* Cache sizing / eviction helpers */

static herr_t H5SC__account_dset_size_change(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, size_t old_size,
                                             size_t new_size);

static herr_t H5SC__account_chunk_link_change(H5SC_t *cache, H5SC_dset_header_t *dset_hdr,
                                              size_t old_dset_size);

/* Freeable-bytes query for two-stage eviction */
static size_t H5SC__calc_freeable_clean_bytes(const H5SC_t *cache);

/* Freeable-bytes query for dirty eviction (flush+evict), bounded by min_dset_size */
static size_t H5SC__calc_freeable_dirty_bytes(const H5SC_t *cache);

/* Verify eviction order reflects true recency (tailmost-eligible policy) */
static herr_t H5SC__verify_candidate_is_tailmost_eligible(const H5SC_dset_header_t *curr_hdr,
                                                          const H5SC_chunk_t *cand_chk, bool allow_dirty);

static bool H5SC__select_evict_candidate_in_dset(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t **cand_chk,
                                                 H5SC_evict_mode_t mode);

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
static herr_t H5SC__ensure_space(H5SC_t *cache, size_t bytes_needed);

static size_t H5SC__calc_inc_for_chunk(const H5D_t *dset, const H5SC_chunk_t *chk, size_t desired);

static herr_t H5SC__invoke_read_dset_batched(H5SC_t *cache, H5D_dset_io_info_t *dset_info,
                                             H5SC_dset_header_t *dset_hdr);
static herr_t H5SC__invoke_write_dset_batched(H5SC_t *cache, H5D_dset_io_info_t *dset_info,
                                              H5SC_dset_header_t *dset_hdr);

/* Dataset specific helper functions */

static herr_t H5SC__dset_exhausted_restore_to_global_lru(H5SC_t *cache, H5SC_exhausted_list_t *exh);
static herr_t H5SC__dset_exhausted_lru_prepend(H5SC_exhausted_list_t     *exh,
                                               struct H5SC_dset_header_t *dset_hdr);
static herr_t H5SC__dset_exhausted_remove(H5SC_exhausted_list_t *exh, struct H5SC_dset_header_t *dset_hdr);

/* Stat-specific functions */
static void   H5SC__stats_reset(H5SC_t *cache);
static void   H5SC__stats_record_lookup(H5SC_t *cache, hbool_t hit);
static void   H5SC__stats_record_insert(H5SC_t *cache);
static void   H5SC__stats_record_delete(H5SC_t *cache);
static void   H5SC__stats_record_chunk_flush(H5SC_t *cache);
static void   H5SC__stats_record_dset_flush(H5SC_t *cache);
static void   H5SC__stats_record_batch_read(H5SC_t *cache);
static void   H5SC__stats_record_batch_write(H5SC_t *cache);
static void   H5SC__stats_record_batch_read_len(H5SC_t *cache, size_t batch_len);
static void   H5SC__stats_record_batch_write_len(H5SC_t *cache, size_t batch_len);
static void   H5SC__stats_record_eviction(H5SC_t *cache);
static double H5SC__stats_get_hit_rate(const H5SC_t *cache);
static double H5SC_stats_get_miss_rate(const H5SC_t *cache);
static herr_t H5SC__stats_dump(const H5SC_t *cache);

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
    #ifdef H5SC_DO_SANITY_CHECKS

        fprintf(stdout, "\n\nH5SC_create(): config_ptr->version = %d.\n", config_ptr->version);
        fprintf(stdout, "               config_ptr->max_q_size = 0x%zu.\n\n",
        (size_t)(config_ptr->max_q_size)); fprintf(stdout, "               config_ptr->max_a_size =
        0x%zu.\n", (size_t)(config_ptr->max_a_size));
    #endif
    */

    /* Allocated cache struct */
    if (NULL == (cache = H5MM_malloc(sizeof(H5SC_t)))) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTALLOC, NULL, "unable to allocate buffer for shared chunk cache");
    }

    /*
     * Initialize cache fields to default values. Some values are available through FAPL configuration;
     * SCC_quiescent_limit - Set by FAPL; if zero, it is set to be 1 GB by default
     * SCC_active_limit - Set by FAPL; if zero, it is set to be 2 GB by default
     */
    cache->SCC_quiescent_size  = (size_t)0;
    cache->SCC_quiescent_limit = config_ptr->max_q_size;
    // cache->SCC_quiescent_limit = (size_t)(1000ULL * 1024ULL * 1024ULL); /* 1 GiB limit by default */
    // cache->SCC_quiescent_limit = (size_t)(1 * 5 * 5 * sizeof(int) + 64); /* DEBUG LIMIT */
    cache->SCC_active_size  = (size_t)0;
    cache->SCC_active_limit = config_ptr->max_a_size;
    // cache->SCC_active_limit = (size_t)(4ULL * cache->SCC_quiescent_limit); /* 4 GiB  limit by default*/
    // cache->SCC_active_limit          = (size_t)(2 * cache->SCC_quiescent_limit); /* DEBUG LIMIT */
    cache->dset_lru_len              = (size_t)0;
    cache->dset_lru_head_ptr         = NULL;
    cache->dset_lru_tail_ptr         = NULL;
    cache->chunk_hash_table_head_ptr = NULL;
    cache->dset_hash_table_head_ptr  = NULL;

    /* Success */
    H5SC__stats_reset(cache);
    ret_value = cache;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_create() */

/*-------------------------------------------------------------------------
 * Function: H5SC_destroy
 *
 * Purpose:  Destroys a shared chunk cache, freeing all data used.
 * Does not flush chunks.
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

#ifdef H5SC_PRINT_STATS
    /* Dump stats before they are reset */

    if (cache->stats.scc_lookups != 0) {
        fprintf(stdout, "\n=== SCC Stats Prior to Cache Destruction ===\n");
        H5SC__stats_dump(cache);
    }

#endif

    H5SC__stats_reset(cache);

    if (cache->dset_lru_len == 0 && cache->SCC_quiescent_size == 0) {

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
 * Purpose:  Flushes all cached data from a shared chunk
 * cache.
 *
 * Return:   SUCCEED on success, FAIL on failure
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_flush(H5SC_t *cache)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    while (cache->dset_lru_tail_ptr) {

        H5SC_dset_header_t *dset_header = cache->dset_lru_tail_ptr;

        if (H5SC__dset_lru_remove(cache, dset_header) < 0) {
            /* Throw an error on fail */
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL,
                        "Failed to remove dataset from the LRU during flush");
        }
        if (H5SC__ht_dset_delete(cache, dset_header->dset_addr) < 0) {
            /* Throw an error on fail */
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL,
                        "Failed to remove dataset from the hash table during flush");
        }
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_flush() */

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
    assert(dset);
    assert(dset_hdr);
    assert(chk);
    assert(dset->shared->layout.sc_ops);
    assert(dset->shared->layout.sc_ops->lookup);
    assert(dset->shared->layout.sc_ops->encode);
    assert(dset->shared->layout.sc_ops->insert);

    /* Shell chunks (non-resident and not on disk) must never be flushed. */
    if (!chk->chunk_obj && !H5_addr_defined(chk->disk_addr)) {
#ifdef H5SC_DO_SANITY_CHECKS
        assert(!chk->dirty_flag);
        assert(chk->cached_chunk_size == 0);
#endif
        HGOTO_DONE(SUCCEED);
    }

    /* A dirty chunk must be resident. */
    if (chk->dirty_flag && !chk->chunk_obj)
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "flush: dirty chunk has no resident object");

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
    chk->dirty_flag  = false;
    H5SC__stats_record_chunk_flush(cache);
    chk->last_op = H5SC_TAG_FLUSH_DIRTY;

done:
    if (write_buf)
        write_buf = H5MM_xfree(write_buf);
    if (udata_arr[0])
        udata_arr[0] = H5MM_xfree(udata_arr[0]);

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
    if (H5SC__chunk_lru_remove(dset_hdr, chk) < 0) {
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
 * Assumed to be present within the SCC.
 *
 * bool evict:
 *      Boolean used to toggle whether data will be be
 * evicted from the cache. This quantity is will be user
 * configurable.
 *
 * Return:   SUCCEED on success, FAIL on failure
 *-------------------------------------------------------------------------
 */
herr_t
H5SC_flush_dset(H5SC_t *cache, H5D_t *dset, bool evict)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *dset_hdr  = NULL;
    H5SC_chunk_t       *chk       = NULL;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
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

        assert(chk->magic == H5SC_CHUNK_MAGIC);

        /* Enforce shell-chunk invariants (H5S_ALL-created, fill-only shells). */
        if (!chk->chunk_obj && !H5_addr_defined(chk->disk_addr)) {
#ifdef H5SC_DO_SANITY_CHECKS
            assert(!chk->dirty_flag);
            assert(chk->cached_chunk_size == 0);
#endif
            if (evict) {
                if (H5SC__evict_one_chunk(cache, dset, dset_hdr, chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed to evict shell chunk (SCC)");
            }
            chk = prev;
            continue;
        }

        /* Flush if dirty */
        if (chk->dirty_flag) {
            if (H5SC__flush_one_chunk(cache, dset, dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTFLUSH, FAIL, "failed to flush dirty chunk (SCC)");
        }

        chk->last_op = H5SC_TAG_FLUSH;

        /* Evict after flushing (or immediately if already clean) */
        if (evict) {
            /* Must be clean to evict */
            chk->dirty_flag = false;

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
    assert(dset_hdr);

    /* Only tracked datasets contribute to global quiescent-size accounting. */
    if (dset_hdr->next_dset_ptr || dset_hdr->prev_dset_ptr || cache->dset_lru_head_ptr == dset_hdr) {
        if (new_size >= old_size)
            cache->SCC_quiescent_size += (new_size - old_size);
        else {
            delta = old_size - new_size;
#ifdef H5SC_DO_SANITY_CHECKS
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
 * Function: H5SC__calc_freeable_clean_bytes
 *
 * Purpose:
 *   Function that computes the maximum number of bytes that could be freed
 *     by evicting clean SCC chunks without reducing any dataset below its
 *     configured minimum retained size (min_dset_size).
 *
 *   The function walks datasets in global LRU order and, within each
 *     dataset, walks chunks in dataset-local LRU order, accumulating the
 *     sizes of evictable clean chunks while enforcing per-dataset minimum
 *     size constraints. Chunks pinned for in-flight I/O are excluded from
 *     the total.
 *
 *   This function is used as the first-stage feasibility check in the
 *     two-stage eviction model, where SCC first determines how much space
 *     can be reclaimed through clean eviction before considering dirty
 *     flush-and-evict candidates.
 *
 * Inputs:
 *   const H5SC_t *cache:
 *     Pointer to the SCC cache state containing the global dataset LRU.
 *
 * Return:
 *   The maximum number of bytes that could be freed by clean eviction
 *   under the current cache and dataset-size constraints.
 *-------------------------------------------------------------------------
 */

static size_t
H5SC__calc_freeable_clean_bytes(const H5SC_t *cache)
{
    size_t freeable = 0;

    assert(cache);

    for (const H5SC_dset_header_t *dset_hdr = cache->dset_lru_tail_ptr; dset_hdr;
         dset_hdr                           = dset_hdr->prev_dset_ptr) {
        size_t cur = dset_hdr->curr_dset_size;
        size_t min = dset_hdr->min_dset_size;

        if (cur <= min)
            continue;

        /* Walk chunks from LRU tail, counting only clean chunks until reaching min */
        for (const H5SC_chunk_t *chk = dset_hdr->lru_tail_ptr; chk; chk = chk->prev_ptr) {
            size_t csz = chk->cached_chunk_size;

            /* Chunks pinned for in-flight I/O are not freeable. */
            if (chk->chunk_counter > 0)
                continue;

            /* Placeholder chunks are not freeable */
            if (!chk->chunk_obj && !H5_addr_defined(chk->disk_addr) && chk->cached_chunk_size == 0)
                continue;

            if (chk->dirty_flag)
                continue;
            if (csz > cur)
                continue;
            if (cur - csz < min)
                continue;
            freeable += csz;
            cur -= csz;
            if (cur <= min)
                break;
        }
    }

    return freeable;
} /* end H5SC__calc_freeable_clean_bytes() */

/*-------------------------------------------------------------------------
 * Function: H5SC__calc_freeable_dirty_bytes
 *
 * Purpose:
 *   Function that computes the maximum number of additional bytes that
 *     could be freed by flushing and evicting dirty SCC chunks without
 *     reducing any dataset below its configured minimum retained size
 *     (min_dset_size).
 *
 *   The function walks datasets in global LRU order and, within each
 *     dataset, walks chunks in dataset-local LRU order while enforcing
 *     per-dataset minimum size constraints. Pinned chunks are excluded.
 *     As the walk consumes evictable chunk sizes against the dataset's
 *     current accounted size, only dirty chunk sizes are accumulated in
 *     the returned total.
 *
 *   This function is used as the second-stage feasibility check in the
 *     two-stage eviction model, after SCC has already determined how much
 *     space can be reclaimed through clean eviction alone.
 *
 * Inputs:
 *   const H5SC_t *cache:
 *     Pointer to the SCC cache state containing the global dataset LRU.
 *
 * Return:
 *   The maximum number of additional bytes that could be freed by
 *   flushing and evicting dirty chunks under the current cache and
 *   dataset-size constraints.
 *-------------------------------------------------------------------------
 */

static size_t
H5SC__calc_freeable_dirty_bytes(const H5SC_t *cache)
{
    size_t freeable = 0;

    assert(cache);

    for (const H5SC_dset_header_t *dset_hdr = cache->dset_lru_tail_ptr; dset_hdr;
         dset_hdr                           = dset_hdr->prev_dset_ptr) {
        size_t cur = dset_hdr->curr_dset_size;
        size_t min = dset_hdr->min_dset_size;

        if (cur <= min)
            continue;

        /* Walk dataset-local LRU from tail while enforcing min_dset_size. */
        for (const H5SC_chunk_t *chk = dset_hdr->lru_tail_ptr; chk; chk = chk->prev_ptr) {
            size_t csz = chk->cached_chunk_size;

            /* Chunks pinned for in-flight I/O are not freeable. */
            if (chk->chunk_counter > 0)
                continue;

            /* Placeholder chunks are not freeable. */
            if (!chk->chunk_obj && !H5_addr_defined(chk->disk_addr) && chk->cached_chunk_size == 0)
                continue;

            if (csz > cur)
                continue;
            if (cur - csz < min)
                continue;

            /* Consume evictable bytes against the dataset minimum regardless of cleanliness. */
            cur -= csz;

            /* Count only dirty bytes as additional reclaimable space. */
            if (chk->dirty_flag) {
                freeable += csz;
            }

            if (cur <= min)
                break;
        }
    }

    return freeable;
} /* end H5SC__calc_freeable_dirty_bytes() */

/*-------------------------------------------------------------------------
 * Function: H5SC__select_evict_candidate_in_dset
 *
 * Purpose:
 *   Select the tailmost eligible eviction candidate from a single dataset.
 *   The search walks the dataset-local chunk LRU from tail to head and
 *   returns the first chunk that satisfies the requested eviction mode and
 *   the dataset's min_dset_size constraint.
 *
 * Inputs:
 *   H5SC_dset_header_t *dset_hdr:
 *     Dataset whose chunk LRU is to be searched.
 *
 *   H5SC_chunk_t **cand_chk:
 *     Output pointer for the selected chunk candidate. Set to NULL if no
 *     eligible chunk is found.
 *
 *   H5SC_evict_mode_t mode:
 *     Eviction mode specifying whether to search for a clean-only or
 *     dirty-only candidate.
 *
 * Return:
 *   true if a matching chunk is found;
 *   false otherwise.
 *-------------------------------------------------------------------------
 */
static bool
H5SC__select_evict_candidate_in_dset(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t **cand_chk,
                                     H5SC_evict_mode_t mode)
{
    size_t cur;
    size_t min;

    assert(dset_hdr);
    assert(cand_chk);

    *cand_chk = NULL;

    /* Dataset must be evictable by callback-driven SCC paths */
    if (!dset_hdr->dset)
        return false;

    if (!dset_hdr->lru_tail_ptr)
        return false;

    cur = dset_hdr->curr_dset_size;
    min = dset_hdr->min_dset_size;

    if (cur <= min)
        return false;

    for (H5SC_chunk_t *chk = dset_hdr->lru_tail_ptr; chk; chk = chk->prev_ptr) {
        size_t csz = chk->cached_chunk_size;

        /* Skip pinned/in-flight chunks */
        if (chk->chunk_counter > 0)
            continue;

        /* Skip placeholder/nonresident chunks */
        if (!chk->chunk_obj && !H5_addr_defined(chk->disk_addr) && chk->cached_chunk_size == 0)
            continue;

        if (mode == H5SC_EVICT_CLEAN_ONLY) {
            if (chk->dirty_flag)
                continue;
        }
        else if (mode == H5SC_EVICT_DIRTY_ONLY) {
            if (!chk->dirty_flag)
                continue;
        }
        else {
            /* Unsupported mode => treat as no candidate */
            return false;
        }

        if (csz > cur)
            continue;

        if ((cur - csz) < min)
            continue;

        *cand_chk = chk;
        return true;
    }

    return false;
} /* end H5SC__select_evict_candidate_in_dset() */

/*-------------------------------------------------------------------------
 * Function: H5SC__verify_candidate_is_tailmost_eligible
 *
 * Purpose:
 *   Debug-time verification function that checks whether a selected SCC
 *     eviction candidate matches the first eligible chunk that would be
 *     found by scanning datasets from the tail of the global dataset LRU
 *     and scanning chunks from the tail of each dataset-local chunk LRU.
 *
 *   Eligibility is recomputed using the same dataset-size and chunk-level
 *     constraints applied during candidate selection. The function is
 *     intended to validate that the chosen candidate is the tailmost
 *     eligible chunk under the current eviction policy.
 *
 * Inputs:
 *   const H5SC_t *cache:
 *     Pointer to the SCC cache state containing the global dataset LRU.
 *
 *   const H5SC_dset_header_t *cand_hdr:
 *     Pointer to the dataset header owning the candidate being verified.
 *
 *   const H5SC_chunk_t *cand_chk:
 *     Pointer to the chunk candidate being verified.
 *
 *   bool allow_dirty:
 *     Flag indicating whether dirty chunks are considered eligible during
 *     verification. If false, only clean chunks are eligible. Should be
 *     changed to an enum if/when H5SC_evict_mode_t supports more than
 *     the current two modes.
 *
 * Return:
 *   SUCCEED if the supplied candidate matches the first eligible chunk
 *   under the current scan order and eligibility rules;
 *   FAIL otherwise.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__verify_candidate_is_tailmost_eligible(const H5SC_dset_header_t *curr_hdr, const H5SC_chunk_t *cand_chk,
                                            bool allow_dirty)
{

    herr_t              ret_value = SUCCEED;
    const H5SC_chunk_t *exp_chk   = NULL;
    size_t              cur;
    size_t              min;

    FUNC_ENTER_PACKAGE

    assert(curr_hdr);
    assert(cand_chk);

    cur = curr_hdr->curr_dset_size;
    min = curr_hdr->min_dset_size;

    if (cur <= min)
        HGOTO_DONE(FAIL);

    if (!curr_hdr->lru_tail_ptr)
        HGOTO_DONE(FAIL);

    for (const H5SC_chunk_t *chk = curr_hdr->lru_tail_ptr; chk; chk = chk->prev_ptr) {
        size_t csz = chk->cached_chunk_size;

        if (chk->chunk_counter > 0)
            continue;

        if (!chk->chunk_obj && !H5_addr_defined(chk->disk_addr) && chk->cached_chunk_size == 0)
            continue;

        if (!allow_dirty && chk->dirty_flag)
            continue;
        if (allow_dirty && !chk->dirty_flag)
            continue;
        if (csz > cur)
            continue;
        if ((cur - csz) < min)
            continue;

        exp_chk = chk;
        break;
    }

    if (!exp_chk)
        HGOTO_ERROR(H5E_SCC, H5E_NOT_TAILMOST, FAIL,
                    "the expected chunk was not found eligible for eviction");

    if (exp_chk != cand_chk)
        HGOTO_ERROR(H5E_SCC, H5E_NOT_TAILMOST, FAIL, "the expected chunk did not match the candidate chunk");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__verify_candidate_is_tailmost_eligible() */

/*-------------------------------------------------------------------------
 * Function: H5SC__trim_to_quiescent_limit
 *
 * Purpose:
 *   Function that reduces SCC quiescent cache usage until
 *     SCC_quiescent_size is less than or equal to
 *     SCC_quiescent_limit. The function repeatedly selects the
 *     tailmost eligible eviction candidate under the current SCC
 *     eviction policy, preferring clean eviction and falling back to
 *     flush-and-evict of dirty chunks when no clean candidate is
 *     available.
 *
 *   Candidate selection respects dataset-local minimum retained size
 *     constraints and dataset/chunk LRU ordering. When a dirty chunk
 *     must be reclaimed, the function first flushes the chunk to disk
 *     and then evicts it from SCC.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state whose quiescent usage is to be
 *     reduced.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__trim_to_quiescent_limit(H5SC_t *cache)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);

    H5SC_exhausted_list_t exh = {0};

    while (cache->SCC_quiescent_size > cache->SCC_quiescent_limit) {
        H5SC_dset_header_t *curr_hdr  = cache->dset_lru_tail_ptr;
        bool                freed_any = false;

        while (curr_hdr && cache->SCC_quiescent_size > cache->SCC_quiescent_limit) {
            H5SC_dset_header_t *next_hdr = curr_hdr->prev_dset_ptr;
            H5SC_chunk_t       *cand_chk = NULL;

            if (!curr_hdr->dset) {
                curr_hdr = next_hdr;
                continue;
            }

            while (cache->SCC_quiescent_size > cache->SCC_quiescent_limit) {
                if (!H5SC__select_evict_candidate_in_dset(curr_hdr, &cand_chk, H5SC_EVICT_CLEAN_ONLY))
                    break;

                cand_chk->last_op = H5SC_TAG_EVICT;
                if (H5SC__evict_one_chunk(cache, curr_hdr->dset, curr_hdr, cand_chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                                "failed to evict chunk during quiescent purge (SCC)");

                freed_any = true;
            }

            while (cache->SCC_quiescent_size > cache->SCC_quiescent_limit) {
                if (!H5SC__select_evict_candidate_in_dset(curr_hdr, &cand_chk, H5SC_EVICT_DIRTY_ONLY))
                    break;

                cand_chk->last_op = H5SC_TAG_FLUSH_DIRTY;
                if (H5SC__flush_one_chunk(cache, curr_hdr->dset, curr_hdr, cand_chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTFLUSH, FAIL,
                                "failed to flush chunk while quiescent purge (SCC)");

                cand_chk->last_op = H5SC_TAG_EVICT;
                if (H5SC__evict_one_chunk(cache, curr_hdr->dset, curr_hdr, cand_chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                                "failed to evict chunk during quiescent purge (SCC)");

                freed_any = true;
            }

            if (H5SC__dset_lru_remove(cache, curr_hdr) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL,
                            "failed to remove exhausted dataset from global LRU");

            if (H5SC__dset_exhausted_lru_prepend(&exh, curr_hdr) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTINSERT, FAIL,
                            "failed to add exhausted dataset to exhausted list");

            curr_hdr = next_hdr;
        }

        if (!freed_any)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "unable to trim quiescent cache further");

        if (H5SC__dset_exhausted_restore_to_global_lru(cache, &exh) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL, "failed to restore exhausted datasets");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__trim_to_quiescent_limit() */

/*-------------------------------------------------------------------------
 * Function: H5SC__ensure_space
 *
 * Purpose:
 *   Function that ensures sufficient SCC cache space is available for a
 *     pending allocation or growth request under SCC_active_limit. If the
 *     request would exceed the active cache limit, the function first
 *     performs a feasibility check and then evicts eligible chunks until
 *     the requested space fits within the limit.
 *
 *   The function follows the SCC two-stage eviction model. First, it
 *     determines whether enough space could be reclaimed to satisfy the
 *     request, preferring clean eviction and considering dirty
 *     flush-and-evict capacity only if clean eviction alone is
 *     insufficient. If the request is feasible, the function then evicts
 *     tailmost eligible chunks in LRU order, preferring clean candidates
 *     and falling back to flushing and evicting dirty candidates when no
 *     clean candidate is available.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the SCC cache state whose active/quiescent accounting is
 *     used for eviction feasibility and space reclamation.
 *
 *   size_t bytes_needed:
 *     Number of additional bytes that must fit within SCC_active_limit.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__ensure_space(H5SC_t *cache, size_t bytes_needed)
{

    herr_t                ret_value = SUCCEED;
    H5SC_exhausted_list_t exh       = {0};

    FUNC_ENTER_PACKAGE

    while (cache->SCC_quiescent_size + bytes_needed > cache->SCC_active_limit) {
        H5SC_dset_header_t *curr_hdr  = cache->dset_lru_tail_ptr;
        bool                freed_any = false;

#ifdef H5SC_DO_SANITY_CHECKS
        printf("Quiescent size: %zu; bytes needed: %zu; active limit: %zu\n", cache->SCC_quiescent_size,
               bytes_needed, cache->SCC_active_limit);
#endif

        /*
         * Stage 2: drain one dataset at a time.
         * For each dataset, evict all eligible clean chunks first, then
         * eligible dirty chunks, before advancing to the next dataset.
         */
        while (curr_hdr && (cache->SCC_quiescent_size + bytes_needed > cache->SCC_active_limit)) {
            H5SC_dset_header_t *next_hdr = curr_hdr->prev_dset_ptr;
            H5SC_chunk_t       *cand_chk = NULL;

            if (!curr_hdr->dset) {
                curr_hdr = next_hdr;
                continue;
            }

            /* Phase 1: drain clean candidates from this dataset. */
            while (cache->SCC_quiescent_size + bytes_needed > cache->SCC_active_limit) {
                if (!H5SC__select_evict_candidate_in_dset(curr_hdr, &cand_chk, H5SC_EVICT_CLEAN_ONLY))
                    break;

#ifdef H5SC_DO_SANITY_CHECKS
                if (H5SC__verify_candidate_is_tailmost_eligible(curr_hdr, cand_chk, false) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                "clean candidate is not tailmost eligible within dataset");
#endif

                cand_chk->last_op = H5SC_TAG_EVICT;
                if (H5SC__evict_one_chunk(cache, curr_hdr->dset, curr_hdr, cand_chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                                "failed to evict clean chunk while draining dataset");

                freed_any = true;
            }

            /* Phase 2: if more space is needed, drain dirty candidates from the same dataset. */
            while (cache->SCC_quiescent_size + bytes_needed > cache->SCC_active_limit) {
                if (!H5SC__select_evict_candidate_in_dset(curr_hdr, &cand_chk, H5SC_EVICT_DIRTY_ONLY))
                    break;

#ifdef H5SC_DO_SANITY_CHECKS
                if (H5SC__verify_candidate_is_tailmost_eligible(curr_hdr, cand_chk, true) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                "dirty candidate is not tailmost eligible within dataset");
#endif

                cand_chk->last_op = H5SC_TAG_FLUSH_DIRTY;
                if (H5SC__flush_one_chunk(cache, curr_hdr->dset, curr_hdr, cand_chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTFLUSH, FAIL,
                                "failed to flush dirty chunk while draining dataset");

                cand_chk->last_op = H5SC_TAG_EVICT;
                if (H5SC__evict_one_chunk(cache, curr_hdr->dset, curr_hdr, cand_chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                                "failed to evict dirty chunk while draining dataset");

                freed_any = true;
            }

            /* This dataset cannot contribute more during this pass. */
            if (H5SC__dset_lru_remove(cache, curr_hdr) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL,
                            "failed to remove exhausted dataset from global LRU");

            if (H5SC__dset_exhausted_lru_prepend(&exh, curr_hdr) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTINSERT, FAIL,
                            "failed to add exhausted dataset to exhausted list");

            curr_hdr = next_hdr;
        }

        if (!freed_any)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "no eviction progress possible while ensuring space");

        if (H5SC__dset_exhausted_restore_to_global_lru(cache, &exh) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTINSERT, FAIL, "failed to restore exhausted datasets to global LRU");
    }

done:
    if (ret_value < 0) {
        if (exh.head || exh.tail || exh.len > 0) {
            if (H5SC__dset_exhausted_restore_to_global_lru(cache, &exh) < 0) {
                /* Best effort only on error path */
            }
        }
    }
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
 *     dataset's chunk size as a conservative upper bound on the required
 *     resident allocation.
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
        /* Use dataset chunk size as a conservative resident-size fallback. */
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
    size_t               processed = 0;

    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(dset_info);
    assert(dset_hdr);
    io = dset_hdr->io_info;
    assert(io);

    saved_sel   = io->sel_chunks;
    saved_start = io->sel_start;
    saved_n     = io->num_sel_chunks;

    if (saved_n == 0)
        HGOTO_DONE(SUCCEED);

#ifdef H5SC_BATCH_INTO_SINGLE_CHUNK
    /* Process each selected chunk individually (debugging mode). */
    size_t i = 0;
    for (i = 0; i < saved_n; i++) {
        io->sel_chunks     = saved_sel;       /* keep base pointer stable */
        io->sel_start      = saved_start + i; /* select subrange start */
        io->num_sel_chunks = 1;

#ifdef DO_SANITY_CHECKS
        H5SC_test_read_batch_calls++;
        H5SC_test_read_last_batch_len = io->num_sel_chunks;
        if (io->num_sel_chunks > H5SC_test_read_max_batch_len)
            H5SC_test_read_max_batch_len = io->num_sel_chunks;
#endif

        if (H5SC_read(cache, 1, dset_info) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to process batched SCC read");
        }

        H5SC__stats_record_batch_read(cache);
        H5SC__stats_record_batch_read_len(cache, 1);

#ifdef H5SC_BATCH_INTO_SINGLE_CHUNK_AND_EVICT_IMMEDIATELY
        /* TEMPORARY: immediately evict the chunk we just processed to avoid
         * eviction pressure / placeholder churn during chunk-by-chunk mode.
         *
         * Note: H5SC__evict_one_chunk() does NOT flush dirty chunks; it asserts
         * clean under sanity checks. H5SC_write() should have already read
         * and marked the chunk clean, but guard defensively.
         */
        {
            const size_t         idx = saved_start + i;
            H5SC_io_sel_chunk_t *sel = &saved_sel[idx];
            H5SC_chunk_t        *chk = sel->cached_chunk;

            if (chk) {
                /* assert(chk->magic == H5SC_CHUNK_MAGIC); */

                if (chk->dirty_flag) {
                    /* For now: do not evict dirty chunks.
                     * Alternative: call your flush helper here, then evict.
                     */
                    /* fprintf(stderr, "skip evict: chunk dirty at idx=%zu\n", idx); */
                }
                else {
                    if (H5SC__evict_one_chunk(cache, dset_info->dset, dset_hdr, chk) < 0)
                        HGOTO_ERROR(H5E_SCC, H5E_CANTREMOVE, FAIL,
                                    "failed to evict chunk after single-chunk read");
                    sel->cached_chunk = NULL; /* optional safety: avoid stale pointer use */
                }
            }
        }
#endif /* H5SC_BATCH_INTO_SINGLE_CHUNK_AND_EVICT_IMMEDIATELY*/
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
        size_t freeable_clean = H5SC__calc_freeable_clean_bytes(cache);
        size_t freeable_dirty = H5SC__calc_freeable_dirty_bytes(cache);
        size_t freeable       = freeable_clean + freeable_dirty;
        size_t budget         = 0;

        if (cache->SCC_quiescent_size >= cache->SCC_active_limit)
            budget = freeable;
        else
            budget = (cache->SCC_active_limit - cache->SCC_quiescent_size) + freeable;

        if (cache->SCC_quiescent_size >= cache->SCC_active_limit)
            budget = freeable;
        else
            budget = (cache->SCC_active_limit - cache->SCC_quiescent_size) + freeable;

        if (budget == 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "invoke read: no budget available");

        /* Start a provisional batch window at the next unprocessed selection. */
        const size_t batch_start = processed;

        /* Configure io_info to expose this provisional window. */
        io->sel_chunks     = saved_sel; /* stable base */
        io->sel_start      = saved_start + batch_start;
        io->num_sel_chunks = saved_n - batch_start; /* provisional; may shrink below */

        /* Populate lookup metadata for the provisional window. */
        if (H5SC__lookup_cache_misses(dset_info->dset, io) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTGET, FAIL, "invoke read: miss lookup caching failed");

        H5SC_io_sel_chunk_t *base = saved_sel;

        /* Window-relevant selection accessor */
#define SEL(j_) (&base[saved_start + batch_start + (j_)])

        size_t batch_len  = 0;
        size_t batch_need = 0;

        /* Greedily extend batch while staying within budget */
        for (size_t j = 0; (batch_start + j) < saved_n; j++) {
            H5SC_io_sel_chunk_t *sel     = SEL(j);
            H5SC_chunk_t        *chk     = sel->cached_chunk;
            size_t               desired = 0;
            size_t               inc     = 0;

            assert(chk);

            /* Estimate additional resident bytes needed for this candidate chunk. */
            if (!chk->chunk_obj) {

                if (sel->lookup_valid)
                    desired = (size_t)sel->lookup_defined_values_size;

                if (desired == 0)
                    desired = (size_t)dset_info->dset->shared->layout.u.struct_chunk.size;

                inc = H5SC__calc_inc_for_chunk(dset_info->dset, chk, desired);
            }

            if (inc > budget)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                            "invoke read: single chunk exceeds effective budget; split further upstream");

            if (batch_need + inc > budget) {
                /* Always make forward progress */
                if (batch_len == 0) {
                    batch_need = inc;
                    batch_len  = 1;
                }
                break;
            }

            batch_need += inc;
            batch_len++;
        }

        assert(batch_len > 0);

        /* Evict before dispatching the batch. batch_need is incremental only. */
        if (batch_need > 0) {
            if (H5SC__ensure_space(cache, batch_need) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                            "invoke read: unable to evict enough space for batch");
        }

        /* Finalize the active window for this batch. */
        io->sel_chunks     = saved_sel;
        io->sel_start      = saved_start + batch_start;
        io->num_sel_chunks = batch_len;

        if (H5SC_read(cache, 1, dset_info) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "invoke read: H5SC_read failed");

        H5SC__stats_record_batch_read(cache);
        H5SC__stats_record_batch_read_len(cache, batch_len);

        processed += batch_len;

#undef SEL
    }

done:

#ifdef H5SC_BATCH_INTO_SINGLE_CHUNK
    /* Unpin any remaining selections after a chunk-by-chunk failure. */
    if (ret_value < 0) {
        size_t start_unpin = (i < saved_n) ? (i + 1) : saved_n;
        for (size_t k = start_unpin; k < saved_n; k++) {
            H5SC_chunk_t *chk2 = saved_sel[saved_start + k].cached_chunk;
            if (chk2 && chk2->chunk_counter > 0) {
                chk2->chunk_counter--; /* decrement exactly once per selection pin */
                chk2->last_op = H5SC_TAG_UNPIN_READ_DONE_ERROR;
            }
        }
    }
#endif

    /* Conservatively unpin any remaining selections after a batched failure. */
    if (ret_value < 0) {
        size_t start_unpin = processed;
        for (size_t k = start_unpin; k < saved_n; k++) {
            H5SC_chunk_t *chk2 = saved_sel[saved_start + k].cached_chunk;
            if (chk2 && chk2->chunk_counter > 0) {
                chk2->chunk_counter--;
                chk2->last_op = H5SC_TAG_UNPIN_READ_DONE_ERROR;
            }
        }
    }

    /* Restore the original selection window. */
    if (dset_hdr && dset_hdr->io_info) {
        dset_hdr->io_info->sel_chunks     = saved_sel;
        dset_hdr->io_info->sel_start      = saved_start;
        dset_hdr->io_info->num_sel_chunks = saved_n;
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
    size_t               processed = 0;

    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(dset_info);
    assert(dset_hdr);
    io = dset_hdr->io_info;
    assert(io);

    saved_sel   = io->sel_chunks;
    saved_start = io->sel_start;
    saved_n     = io->num_sel_chunks;

    if (saved_n == 0)
        HGOTO_DONE(SUCCEED);

#ifdef H5SC_BATCH_INTO_SINGLE_CHUNK
    /* Process each selected chunk individually (debugging mode). */
    size_t i = 0;
    for (i = 0; i < saved_n; i++) {
        io->sel_chunks     = saved_sel;       /* keep base pointer stable */
        io->sel_start      = saved_start + i; /* select subrange start */
        io->num_sel_chunks = 1;

#ifdef DO_SANITY_CHECKS
        H5SC_test_write_batch_calls++;
        H5SC_test_write_last_batch_len = io->num_sel_chunks;
        if (io->num_sel_chunks > H5SC_test_write_max_batch_len)
            H5SC_test_write_max_batch_len = io->num_sel_chunks;
#endif

        if (H5SC_write(cache, 1, dset_info) < 0) {
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to process batched SCC write");
        }

        H5SC__stats_record_batch_write(cache);
        H5SC__stats_record_batch_read_len(cache, 1);

#ifdef H5SC_BATCH_INTO_SINGLE_CHUNK_AND_EVICT_IMMEDIATELY
        /* TEMPORARY: immediately evict the chunk we just processed to avoid
         * eviction pressure / placeholder churn during chunk-by-chunk mode.
         *
         * Note: H5SC__evict_one_chunk() does NOT flush dirty chunks; it asserts
         * clean under sanity checks. H5SC_write() should have already written
         * and marked the chunk clean, but guard defensively.
         */
        {
            const size_t         idx = saved_start + i;
            H5SC_io_sel_chunk_t *sel = &saved_sel[idx];
            H5SC_chunk_t        *chk = sel->cached_chunk;

            if (chk) {
                /* assert(chk->magic == H5SC_CHUNK_MAGIC); */

                if (chk->dirty_flag) {
                    /* For now: do not evict dirty chunks.
                     * Alternative: call your flush helper here, then evict.
                     */
                    /* fprintf(stderr, "skip evict: chunk dirty at idx=%zu\n", idx); */
                }
                else {

                    if (chk->chunk_counter > 0)
                        HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                                    "post-write eviction attempted on a pinned chunk");

                    if (H5SC__evict_one_chunk(cache, dset_info->dset, dset_hdr, chk) < 0)
                        HGOTO_ERROR(H5E_SCC, H5E_CANTREMOVE, FAIL,
                                    "failed to evict chunk after single-chunk write");
                }
            }
        }
#endif /* H5SC_BATCH_INTO_SINGLE_CHUNK_AND_EVICT_IMMEDIATELY */
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
        size_t freeable_clean = H5SC__calc_freeable_clean_bytes(cache);
        size_t freeable_dirty = H5SC__calc_freeable_dirty_bytes(cache);
        size_t freeable       = freeable_clean + freeable_dirty;
        size_t budget         = 0;

        if (cache->SCC_quiescent_size >= cache->SCC_active_limit)
            budget = freeable;
        else
            budget = (cache->SCC_active_limit - cache->SCC_quiescent_size) + freeable;

        if (cache->SCC_quiescent_size >= cache->SCC_active_limit)
            budget = freeable;
        else
            budget = (cache->SCC_active_limit - cache->SCC_quiescent_size) + freeable;

        if (budget == 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "invoke write: no budget available");

        /* Start a provisional batch window at the next unprocessed selection. */
        const size_t batch_start = processed;

        /* Configure io_info to expose this provisional window. */
        io->sel_chunks     = saved_sel; /* stable base */
        io->sel_start      = saved_start + batch_start;
        io->num_sel_chunks = saved_n - batch_start; /* provisional; may shrink below */

        /* Populate lookup metadata for the provisional window. */
        if (H5SC__lookup_cache_misses(dset_info->dset, io) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTGET, FAIL, "invoke write: miss lookup caching failed");

        H5SC_io_sel_chunk_t *base = saved_sel;

        /* Window-relevant selection accessor */
#define SEL(j_) (&base[saved_start + batch_start + (j_)])

        size_t batch_len  = 0;
        size_t batch_need = 0;

        /* Greedily extend batch while staying within budget */
        for (size_t j = 0; (batch_start + j) < saved_n; j++) {
            H5SC_io_sel_chunk_t *sel     = SEL(j);
            H5SC_chunk_t        *chk     = sel->cached_chunk;
            size_t               desired = 0;
            size_t               inc     = 0;

            assert(chk);

            /*
             * We only need additional budget when the chunk has no resident decoded object yet.
             * Use lookup_size_hint when available; fall back to full chunk size for sparse/unallocated.
             * NOTE: lookup_size_hint is a very pessimistic estimate and assumes the chunk is dense.
             *       It may be worth setting the desired size based on the number of points in the dataset,
             *       and when lookup_valid is false, defaulting to the worst-case estimate.
             */
            if (!chk->chunk_obj) {
                if (sel->lookup_valid)
                    desired = (size_t)sel->lookup_defined_values_size;

                if (desired == 0)
                    desired = (size_t)dset_info->dset->shared->layout.u.struct_chunk.size;

                /*
                 * IMPORTANT: keep budget semantics identical to the core SCC code.
                 * This handles placeholder/shell accounting correctly.
                 */
                inc = H5SC__calc_inc_for_chunk(dset_info->dset, chk, desired);
            }

            if (inc > budget) {
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                            "invoke write: single chunk exceeds effective budget; split further upstream");
            }

            if (batch_need + inc > budget) {
                /* Always make forward progress (even if it means a 1-chunk batch) */
                if (batch_len == 0) {
                    batch_need = inc;
                    batch_len  = 1;
                }
                break;
            }

            batch_need += inc;
            batch_len++;
        }

        assert(batch_len > 0);

        /* Evict before dispatching the batch. batch_need is incremental only. */
        if (batch_need > 0) {
            if (H5SC__ensure_space(cache, batch_need) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL,
                            "invoke write: unable to evict enough space for batch");
        }

        /* Finalize the active window for this batch. */
        io->sel_chunks     = saved_sel;                 /* stable base */
        io->sel_start      = saved_start + batch_start; /* absolute start in base */
        io->num_sel_chunks = batch_len;

#ifdef DO_SANITY_CHECKS
        H5SC_test_write_batch_calls++;
        H5SC_test_write_last_batch_len = io->num_sel_chunks;
        if (io->num_sel_chunks > H5SC_test_write_max_batch_len)
            H5SC_test_write_max_batch_len = io->num_sel_chunks;
#endif

        if (H5SC_write(cache, 1, dset_info) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "invoke write: H5SC_write failed");

        H5SC__stats_record_batch_write(cache);
        H5SC__stats_record_batch_write_len(cache, batch_len);

        processed += batch_len;

#undef SEL
    }

done:

#ifdef H5SC_BATCH_INTO_SINGLE_CHUNK
    /* Unpin any remaining selections after a chunk-by-chunk failure. */
    if (ret_value < 0) {
        size_t start_unpin = (i < saved_n) ? (i + 1) : saved_n;
        for (size_t k = start_unpin; k < saved_n; k++) {
            H5SC_chunk_t *chk2 = saved_sel[saved_start + k].cached_chunk;
            if (chk2 && chk2->chunk_counter > 0) {
                chk2->chunk_counter--; /* decrement exactly once per selection pin */
                chk2->last_op = H5SC_TAG_UNPIN_WRITE_DONE_ERROR;
            }
        }
    }
#endif

    /* Conservatively unpin any remaining selections after a batched failure. */
    if (ret_value < 0) {
        size_t start_unpin = processed;
        for (size_t k = start_unpin; k < saved_n; k++) {
            H5SC_chunk_t *chk2 = saved_sel[saved_start + k].cached_chunk;
            if (chk2 && chk2->chunk_counter > 0) {
                chk2->chunk_counter--; /* exactly once per selection pin */
                chk2->last_op = H5SC_TAG_UNPIN_WRITE_DONE_ERROR;
            }
        }
    }

    /* Restore the original selection window. */
    if (dset_hdr && dset_hdr->io_info) {
        dset_hdr->io_info->sel_chunks     = saved_sel;
        dset_hdr->io_info->sel_start      = saved_start;
        dset_hdr->io_info->num_sel_chunks = saved_n;
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
    H5SC_io_sel_chunk_t *base          = NULL;
    size_t               miss_count    = 0;
    size_t               sel_start     = 0;
    haddr_t              md_tag        = HADDR_UNDEF;
    herr_t               ret_value     = SUCCEED;
    const hsize_t      **scaled_miss   = NULL;
    haddr_t            **addr_miss     = NULL;
    hsize_t            **size_miss     = NULL;
    hsize_t            **def_sz_miss   = NULL;
    size_t             **hint_miss     = NULL;
    size_t             **def_hint_miss = NULL;
    void               **udata_miss    = NULL;
    size_t              *miss_idx      = NULL;

    FUNC_ENTER_PACKAGE

    assert(dset);
    assert(sc_io_info);
    assert(dset->shared->layout.sc_ops->lookup);

    sel_start = sc_io_info->sel_start;
    base      = sc_io_info->sel_chunks;

/* Selection accessor for the active window. */
#define SEL(j_) (&base[sel_start + (j_)])

    /* Count non-resident selections that still need lookup metadata. */
    for (size_t j = 0; j < sc_io_info->num_sel_chunks; j++) {

        H5SC_io_sel_chunk_t *sel = SEL(j);
        H5SC_chunk_t        *chk = sel->cached_chunk;
        assert(chk);

        if (chk->chunk_obj)
            continue; /* resident => no lookup needed */

        if (sel->lookup_valid == true)
            continue; /* already cached for this invoke */

        miss_count++;
    }

    if (miss_count == 0) {
        HGOTO_DONE(SUCCEED);
    }

    /* Allocate lookup scratch arrays on the heap. */
    if (NULL == (scaled_miss = (const hsize_t **)H5MM_malloc(miss_count * sizeof(*scaled_miss))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate scaled_miss array");
    if (NULL == (addr_miss = (haddr_t **)H5MM_malloc(miss_count * sizeof(*addr_miss))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate addr_miss array");
    if (NULL == (size_miss = (hsize_t **)H5MM_malloc(miss_count * sizeof(*size_miss))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate size_miss array");
    if (NULL == (def_sz_miss = (hsize_t **)H5MM_malloc(miss_count * sizeof(*def_sz_miss))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate def_sz_miss array");
    if (NULL == (hint_miss = (size_t **)H5MM_malloc(miss_count * sizeof(*hint_miss))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate hint_miss array");
    if (NULL == (def_hint_miss = (size_t **)H5MM_malloc(miss_count * sizeof(*def_hint_miss))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate def_hint_miss array");
    if (NULL == (udata_miss = (void **)H5MM_malloc(miss_count * sizeof(*udata_miss))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate udata_miss array");
    if (NULL == (miss_idx = (size_t *)H5MM_malloc(miss_count * sizeof(*miss_idx))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate miss_idx array");

    /* Build lookup argument arrays for the current miss set. */
    {
        size_t k = 0;

        for (size_t j = 0; j < sc_io_info->num_sel_chunks; j++) {
            H5SC_io_sel_chunk_t *sel = SEL(j);
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

            udata_miss[k] = NULL;
            k++;
        }

        /* Tag metadata accesses for the batched lookup. */
        H5AC_tag(dset->oloc.addr, &md_tag);

        if (dset->shared->layout.sc_ops->lookup(dset, miss_count, scaled_miss, addr_miss, size_miss,
                                                def_sz_miss, hint_miss, def_hint_miss, udata_miss) < 0) {
            H5AC_tag(md_tag, NULL); /* Reset the metadata tag on error as well */
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to lookup chunk (SCC)");
        }

        H5AC_tag(md_tag, NULL); /* Reset the metadata tag for the next dataset */

        /* Commit returned lookup metadata. */
        for (size_t k2 = 0; k2 < miss_count; k2++) {
            size_t               j   = miss_idx[k2];
            H5SC_io_sel_chunk_t *sel = SEL(j);
            H5SC_chunk_t        *chk = sel->cached_chunk;

            sel->lookup_udata = udata_miss[k2];
            sel->lookup_valid = true;

            if (chk && !chk->udata)
                chk->udata = udata_miss[k2];
        }
    }

#undef SEL
done:
    scaled_miss   = (const hsize_t **)H5MM_xfree((void *)scaled_miss);
    addr_miss     = (haddr_t **)H5MM_xfree(addr_miss);
    size_miss     = (hsize_t **)H5MM_xfree(size_miss);
    def_sz_miss   = (hsize_t **)H5MM_xfree(def_sz_miss);
    hint_miss     = (size_t **)H5MM_xfree(hint_miss);
    def_hint_miss = (size_t **)H5MM_xfree(def_hint_miss);
    udata_miss    = (void **)H5MM_xfree(udata_miss);
    miss_idx      = (size_t *)H5MM_xfree(miss_idx);

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

#ifdef H5SC_DO_SANITY_CHECKS
        /* Sanity check to ensure the is present in */
        H5SC_dset_header_t *dset_hdr = H5SC__ht_dset_find(cache, dset_info[i].dset->oloc.addr);
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
                    sel_chunk->cached_chunk->chunk_counter++;
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

                    if (H5SC__chunk_lru_prepend(dset_hdr, cached_chk) < 0) {
                        HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL,
                                    "failed to insert chunk into dset dll during io init process");
                    }
                    /* Increase pin count to retain this chunk during mid-I/O request processing evictions */
                    cached_chk->chk_log_coord = log_chk_idx;
                    cached_chk->chunk_counter++;
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
    }

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
 * Function: H5SC__erase_io_info_init
 *
 * Purpose:
 *   Initialize SCC per-request state for an erase operation.
 *
 *   This is a trimmed variant of H5SC__io_info_init() specialized for
 *   H5SC_erase().  It identifies the chunks intersecting file_space,
 *   finds or creates SCC chunk entries for them, pins those entries for
 *   the duration of the erase operation, and computes the per-chunk
 *   file-space selection in logical chunk coordinates.
 *
 *   Unlike H5SC__io_info_init(), this function does not compute or store
 *   per-chunk memory-space selections because erase does not involve
 *   scatter/gather operations.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */
static herr_t
H5SC__erase_io_info_init(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space)
{
    H5SC_dset_header_t *dset_hdr           = NULL;
    H5SC_io_info_t     *sc_io_info         = NULL;
    H5S_t              *single_chunk_space = NULL;
    hsize_t             chunk_dims[H5S_MAX_RANK];
    hsize_t             file_dims[H5S_MAX_RANK];
    hsize_t             start_coords[H5O_LAYOUT_NDIMS];
    hsize_t             coords[H5S_MAX_RANK];
    hsize_t             end[H5S_MAX_RANK];
    hsize_t             start_scaled[H5S_MAX_RANK];
    hsize_t             scaled[H5S_MAX_RANK];
    hsize_t             file_sel_start[H5S_MAX_RANK];
    hsize_t             file_sel_end[H5S_MAX_RANK];
    hsize_t             zeros[H5S_MAX_RANK];
    hsize_t             curr_partial_clip[H5S_MAX_RANK];
    hsize_t             partial_dim_size[H5S_MAX_RANK];
    bool                is_partial_dim[H5S_MAX_RANK] = {false};
    hsize_t             sel_points;
    hsize_t             dset_sel_chunks = 0;
    unsigned            file_ndims;
    unsigned            num_partial_dims = 0;
    H5S_sel_type        file_sel_type;
    int                 curr_dim;
    unsigned            u;
    herr_t              ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(dset);
    assert(file_space);
    assert(dset->shared->layout.sc_ops);
    assert(dset->shared->layout.sc_ops->layout_query);

    dset_hdr = H5SC__ht_dset_find(cache, dset->oloc.addr);
    if (dset_hdr == NULL)
        HGOTO_ERROR(H5E_SCC, H5E_HT_NOTFOUND, FAIL,
                    "dataset should have been added to the hash table upon creation");

    dset_hdr->dset = dset;

    sc_io_info = dset_hdr->io_info;
    assert(sc_io_info);

    if (H5SC__io_info_reset(sc_io_info) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_IOFAIL, FAIL, "failed to reset sc_io_info during erase init");

    sel_points = H5S_GET_SELECT_NPOINTS(file_space);
    if (sel_points == 0)
        HGOTO_DONE(SUCCEED);

    if (dset->shared->layout.sc_ops->layout_query(dset, chunk_dims, NULL, NULL) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to query chunk dimensions");

    file_ndims = (unsigned)H5S_GET_EXTENT_NDIMS(file_space);

    if ((file_sel_type = H5S_GET_SELECT_TYPE(file_space)) < H5S_SEL_NONE)
        HGOTO_ERROR(H5E_DATASET, H5E_BADSELECT, FAIL, "unable to get type of selection");

    if (H5S_get_simple_extent_dims(file_space, file_dims, NULL) < 0)
        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL, "can't get file dataspace dimensions");

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

        if (NULL == (single_chunk_space = H5S_create_simple(file_ndims, chunk_dims, NULL)))
            HGOTO_ERROR(H5E_DATASET, H5E_CANTCREATE, FAIL, "unable to create dataspace for chunk");
    }
    else {
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

    while (sel_points) {
        if ((H5S_SEL_ALL == file_sel_type) || (true == H5S_SELECT_INTERSECT_BLOCK(file_space, coords, end))) {
            H5SC_io_sel_chunk_t *sel_chunk   = NULL;
            hsize_t              log_chk_idx = 0;
            H5SC_chunk_key_t     chk_key;
            H5SC_chunk_t        *cached_chk = NULL;

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

            sel_chunk = &(sc_io_info->sel_chunks[sc_io_info->num_sel_chunks]);

            H5SC__io_sel_chunk_init(sel_chunk);

            if (H5SC__compute_logical_chunk_index(file_ndims, file_dims, chunk_dims, coords, &log_chk_idx) <
                0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTCOMPUTE, FAIL, "failed to compute linear chunk index");

            sel_chunk->chk_log_coord = log_chk_idx;

            if (H5SC__compute_chunk_key(&(dset->oloc.addr), &(sel_chunk->chk_log_coord), &chk_key) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CHUNK_KEY, FAIL, "failed to compute chunk key during erase init");

            cached_chk = H5SC__ht_chunk_find(cache, &chk_key);

            if (cached_chk) {
                cached_chk->chk_log_coord = log_chk_idx;
                sel_chunk->cached_chunk   = cached_chk;
                sel_chunk->cached_chunk->chunk_counter++;
                sel_chunk->cached_chunk->last_op = H5SC_TAG_PIN_IOINIT_CACHED;
            }
            else {
                cached_chk = H5SC__make_chunk(chk_key, (size_t)0, (size_t)0, false);
                if (!cached_chk)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTALLOC, FAIL, "failed to create chunk during erase init");

                if (H5SC__ht_chunk_insert(cache, cached_chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL,
                                "failed to insert chunk into hash table during erase init");

                if (H5SC__chunk_lru_prepend(dset_hdr, cached_chk) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANNOTINSERT, FAIL,
                                "failed to insert chunk into dataset LRU during erase init");

                cached_chk->chk_log_coord = log_chk_idx;
                cached_chk->chunk_counter++;
                cached_chk->last_op     = H5SC_TAG_PIN_IOINIT;
                sel_chunk->cached_chunk = cached_chk;
            }

            sc_io_info->num_sel_chunks++;

            if (H5S_SEL_ALL == file_sel_type) {
                if (NULL == (sel_chunk->file_space = H5S_copy(single_chunk_space, true, false)))
                    HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL, "unable to copy chunk dataspace");

                if (num_partial_dims > 0)
                    if (H5S_select_hyperslab(sel_chunk->file_space, H5S_SELECT_SET, zeros, NULL,
                                             curr_partial_clip, NULL) < 0)
                        HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "can't create chunk erase selection");
            }
            else {
                if (H5S_SEL_HYPERSLABS == file_sel_type) {
                    hsize_t clip_count[H5S_MAX_RANK];

                    for (u = 0; u < file_ndims; u++) {
                        if (coords[u] >= file_dims[u])
                            HGOTO_ERROR(
                                H5E_SCC, H5E_BADVALUE, FAIL,
                                "attempted to build erase selection for chunk outside dataset extent");

                        clip_count[u] = MIN(chunk_dims[u], file_dims[u] - coords[u]);

                        if (clip_count[u] == 0)
                            HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                        "computed zero-sized clipped erase chunk selection");
                    }

                    if (H5S_combine_hyperslab(file_space, H5S_SELECT_AND, coords, NULL, clip_count, NULL,
                                              &sel_chunk->file_space) < 0)
                        HGOTO_ERROR(H5E_DATASPACE, H5E_CANTCOPY, FAIL,
                                    "unable to combine erase selection with clipped chunk block");
                }
                else {
                    HGOTO_ERROR(H5E_DATASPACE, H5E_UNSUPPORTED, FAIL,
                                "point selections are not yet supported (SCC erase)");
                }

                if (H5S_set_extent_real(sel_chunk->file_space, chunk_dims) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "can't adjust chunk dimensions");

                if (H5S_SELECT_ADJUST_U(sel_chunk->file_space, coords) < 0)
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTSELECT, FAIL, "can't adjust chunk selection");
            }

            hsize_t chunk_points = H5S_GET_SELECT_NPOINTS(sel_chunk->file_space);

            if (chunk_points == 0 || chunk_points > sel_points)
                HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                            "invalid selected chunk point count during SCC erase init");

            sel_points -= chunk_points;

            dset_sel_chunks++;
        }

        curr_dim = (int)file_ndims - 1;

        coords[curr_dim] += chunk_dims[curr_dim];
        end[curr_dim] += chunk_dims[curr_dim];
        scaled[curr_dim]++;

        if (coords[curr_dim] > file_sel_end[curr_dim]) {
            do {
                scaled[curr_dim] = start_scaled[curr_dim];
                coords[curr_dim] = start_coords[curr_dim];
                end[curr_dim]    = (coords[curr_dim] + chunk_dims[curr_dim]) - 1;

                if (is_partial_dim[curr_dim] && end[curr_dim] < file_dims[curr_dim]) {
                    assert(num_partial_dims > 0);
                    assert(H5S_SEL_ALL == file_sel_type);

                    curr_partial_clip[curr_dim] = chunk_dims[curr_dim];
                    is_partial_dim[curr_dim]    = false;
                    num_partial_dims--;
                }

                curr_dim--;

                if (curr_dim >= 0) {
                    scaled[curr_dim]++;
                    coords[curr_dim] += chunk_dims[curr_dim];
                    end[curr_dim] = (coords[curr_dim] + chunk_dims[curr_dim]) - 1;
                }
            } while (curr_dim >= 0 && (coords[curr_dim] > file_sel_end[curr_dim]));

            if ((H5S_SEL_ALL == file_sel_type) && curr_dim >= 0) {
                if (!is_partial_dim[curr_dim] && file_dims[curr_dim] <= end[curr_dim]) {
                    curr_partial_clip[curr_dim] = partial_dim_size[curr_dim];
                    is_partial_dim[curr_dim]    = true;
                    num_partial_dims++;
                    assert(num_partial_dims <= file_ndims);
                }
            }
        }
    }

done:
    if (ret_value < 0) {
        if (sc_io_info) {
            if (H5SC__io_info_reset(sc_io_info) < 0)
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't close within erase I/O info");
        }
    }

    if (single_chunk_space && H5S_close(single_chunk_space) < 0)
        HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL,
                    "can't release temporary chunk dataspace during erase init");

    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__erase_io_info_init() */

/*-------------------------------------------------------------------------
 * Function: H5SC__io_info_reset
 *
 * Purpose:
 *   Function that resets an H5SC_io_info_t structure for reuse after a
 *     dataset I/O operation. The function releases any temporary per-chunk
 *     dataspaces owned by the structure, clears per-chunk transient state,
 *     and resets the active selected-chunk count while preserving the
 *     underlying sel_chunks allocation for later reuse.
 *
 *   In addition to clearing the currently active chunk-selection entries,
 *     the function resets all per-invocation lookup-related fields across
 *     the full allocated sel_chunks array so that stale lookup metadata is
 *     not reused by subsequent I/O operations.
 *
 * Inputs:
 *   H5SC_io_info_t *sc_io_info:
 *     Pointer to the I/O request state structure to reset.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
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
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary file dataspace");

        if (c->mem_space && !c->mem_space_shared)
            if (H5S_close(c->mem_space) < 0)
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary memory dataspace");

        c->file_space        = NULL;
        c->mem_space         = NULL;
        c->file_space_shared = false;
        c->mem_space_shared  = false;
        c->dset_info         = NULL;
    }

    /* Mark the active selection set as empty. */
    sc_io_info->num_sel_chunks = 0;

    /* Clear per-invocation lookup metadata across teh full allocation */
    if (sc_io_info->sel_chunks) {
        for (size_t i = 0; i < sc_io_info->sel_chunks_alloced; i++) {
            sc_io_info->sel_chunks[i].lookup_valid                    = false;
            sc_io_info->sel_chunks[i].lookup_addr                     = HADDR_UNDEF;
            sc_io_info->sel_chunks[i].lookup_disk_nbytes              = 0;
            sc_io_info->sel_chunks[i].lookup_defined_values_size      = 0;
            sc_io_info->sel_chunks[i].lookup_size_hint                = 0;
            sc_io_info->sel_chunks[i].lookup_defined_values_size_hint = 0;
            sc_io_info->sel_chunks[i].lookup_udata                    = NULL;
        }
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__io_info_reset()*/

/*-------------------------------------------------------------------------
 * Function: H5SC__io_info_term
 *
 * Purpose:
 *   Function that releases resources owned by an H5SC_io_info_t structure
 *     and returns it to an uninitialized/empty state. The function closes
 *     any owned temporary per-chunk dataspaces for the currently active
 *     selected chunks, frees the sel_chunks backing array, and clears the
 *     structure’s bookkeeping fields.
 *
 *   This function releases memory and transient handles referenced by
 *     sc_io_info, but does not free the H5SC_io_info_t structure itself.
 *
 * Inputs:
 *   H5SC_io_info_t *sc_io_info:
 *     Pointer to the I/O request state structure to tear down.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
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
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary file dataspace");

        if (sc_io_info->sel_chunks[i].mem_space && !sc_io_info->sel_chunks[i].mem_space_shared)
            if (H5S_close(sc_io_info->sel_chunks[i].mem_space) < 0)
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't release temporary memory dataspace");
    }

    /* Free backing storage and reset structure state. */
    sc_io_info->sel_chunks         = (H5SC_io_sel_chunk_t *)H5MM_xfree(sc_io_info->sel_chunks);
    sc_io_info->num_sel_chunks     = 0;
    sc_io_info->sel_chunks_alloced = 0;
    sc_io_info->sel_start          = 0;

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
 *   size_t count:
 *     Number of dataset I/O entries in dset_info. The current SCC read
 *     path expects count == 1. (TO BE REMOVED)
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
H5SC_read(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info)
{

    herr_t             ret_value = SUCCEED;
    H5D_io_type_info_t my_io_type_info;    /* First used in the scatter_mem callback */
    const H5S_t       *scatter_mem_space;  /* Used in the scatter_mem callback */
    const H5S_t       *scatter_file_space; /* Used in the scatter_mem callback */
    haddr_t            md_tag                                  = HADDR_UNDEF;
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
    assert(count == 0 || dset_info);
    assert(count == 1);

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

    chunk_count = sc_io_info->num_sel_chunks;

    /* Allocate scratch arrays used for lookup/read processing. */
    if (NULL == (scaled = (const hsize_t **)H5MM_malloc(chunk_count * sizeof(*scaled))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate scaled array");

    if (NULL == (addr = (haddr_t **)H5MM_malloc(chunk_count * sizeof(*addr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate addr array");

    if (NULL == (size = (hsize_t **)H5MM_malloc(chunk_count * sizeof(*size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate size array");

    if (NULL == (defined_values_size = (hsize_t **)H5MM_malloc(chunk_count * sizeof(*defined_values_size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate defined_values_size array");

    if (NULL == (size_hint = (size_t **)H5MM_malloc(chunk_count * sizeof(*size_hint))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate size_hint array");

    if (NULL ==
        (defined_values_size_hint = (size_t **)H5MM_malloc(chunk_count * sizeof(*defined_values_size_hint))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate defined_values_size_hint array");

    if (NULL == (udata_arr = (void **)H5MM_malloc(chunk_count * sizeof(*udata_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate udata_arr array");

    if (NULL == (miss_arr = (size_t *)H5MM_malloc(chunk_count * sizeof(*miss_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate miss_arr array");

    if (NULL == (chunk_arr = (void **)H5MM_malloc(chunk_count * sizeof(*chunk_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate chunk_arr array");

        /* Window-relevant selection accessor */
#define SEL(j_) (&base[sel_start + (j_)])
    /* Chunk lookup (in file) */

    /* For each chunk in this dataset, initialize each necessary array value*/
    for (size_t j = 0; j < chunk_count; j++) {
        /* Setup scaled for the jth chunk */
        scaled[j] = SEL(j)->scaled;

        /* Setup the initial address with a unique pointer */
        haddr_t *tmp_addr = H5MM_malloc(sizeof(haddr_t));
        *tmp_addr         = HADDR_UNDEF;
        addr[j]           = tmp_addr;

        /* Setup the initial size with a unique pointer */
        hsize_t *tmp_size = H5MM_malloc(sizeof(hsize_t));
        *tmp_size         = 0;
        size[j]           = tmp_size;

        hsize_t *tmp_def_val_size = H5MM_malloc(sizeof(hsize_t));
        *tmp_def_val_size         = 0;
        defined_values_size[j]    = tmp_def_val_size;

        size_t *tmp_size_hint = H5MM_malloc(sizeof(size_t));
        *tmp_size_hint        = 0;
        size_hint[j]          = tmp_size_hint;

        size_t *tmp_def_val_size_hint = H5MM_malloc(sizeof(size_t));
        *tmp_def_val_size_hint        = 0;
        defined_values_size_hint[j]   = tmp_def_val_size_hint;

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

        H5SC_chunk_t *chk = SEL(j)->cached_chunk;

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

        H5SC_io_sel_chunk_t *sel = SEL(idx);
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

        my_io_type_info.tconv_buf      = NULL; /* Pointer to the datatype conv buffer */
        my_io_type_info.tconv_buf_size = dset_info[0].type_info.src_type_size;
        my_io_type_info.bkg_buf        = NULL; /* Pointer to background buffer */
        my_io_type_info.bkg_buf_size   = dset_info[0].type_info.dst_type_size;

        sel_chunk    = SEL(j);
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
            if (!(SEL(j)->cached_chunk && SEL(j)->cached_chunk->udata == udata_arr[j])) {
                udata_arr[j] = H5MM_xfree(udata_arr[j]);
            }

            /* Create a new chunk that will then have the fill value written to it. */
            if (dset_info[0].dset->shared->layout.sc_ops->new_chunk(
                    dset_info[0].dset, false, size[j], size_hint[j], &chunk_arr[j], &udata_arr[j]) < 0) {
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to create new chunk (SCC)");
            }

            if (dset_info[0].dset->shared->layout.sc_ops->fill(
                    &dset_info[0], &my_io_type_info, SEL(j)->file_space, size[j], size_hint[j],
                    &alloc_size_total, chunk_arr[j], udata_arr[j]) < 0) {
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to fill chunk (SCC)");
            }
            /* Add info for the resident chunk within the cache */
            cached_chunk->chunk_obj   = chunk_arr[j];
            cached_chunk->udata       = udata_arr[j];
            cached_chunk->disk_addr   = HADDR_UNDEF;
            cached_chunk->disk_nbytes = 0;
            cached_chunk->dirty_flag  = false;

            /* Update per-dataset bytes now; global accounting is reconciled once per request */
            if (H5SC__chunk_update_cached_size(dset_hdr, cached_chunk, *size[j]) < 0) {
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "read: chunk size accounting failed");
            }
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

            /* Skip for invalid addr */
            if (dset_info[0].dset->shared->layout.sc_ops->decode(dset_info[0].dset, size[j], size_hint[j],
                                                                 partial_bound, &chunk_arr[j],
                                                                 udata_arr[j]) < 0) {
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to decode chunk in place (SCC)");
            }

            /* Add info for the resident chunk within the cache */
            cached_chunk->chunk_obj  = chunk_arr[j];
            cached_chunk->udata      = udata_arr[j];
            cached_chunk->disk_addr  = *addr[j];
            cached_chunk->dirty_flag = false;

            /* Update LRU byte counters + cache quiescent bytes */
            if (H5SC__chunk_update_cached_size(dset_hdr, cached_chunk, *size[j]) < 0) {
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "chunk size accounting failed (SCC read)");
            }
        }

        if (dset_info[0].dset->shared->layout.sc_ops->scatter_mem(&dset_info[0], &my_io_type_info,
                                                                  scatter_mem_space, scatter_file_space,
                                                                  chunk_arr[j], udata_arr[j]) < 0) {
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to scatter mem for read chunk (SCC)");
        }

        {
            H5SC_chunk_t *chk = SEL(j)->cached_chunk;
            assert(chk);

            if (chk->chunk_counter > 0) {
                chk->chunk_counter--;
                chk->last_op = H5SC_TAG_UNPIN_READ_DONE;
            }
            else {
                HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                            "SCC read: chunk counter was not properly incremented");
            }
        }

    } /* Chunk Processing Loop End */

    /* Reconcile dataset/global cache size accounting once per request */
    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "global size accounting failed (SCC read)");
    }

    H5AC_tag(md_tag, NULL); /* Reset the metadata tag for the next dataset */
#undef SEL

done:

    if (ret_value < 0) {
        if (sc_io_info && base && sel_count > 0) {
            for (size_t j = 0; j < sel_count; j++) {
                H5SC_chunk_t *chk = base[sel_start + j].cached_chunk;

                if (chk && chk->chunk_counter > 0) {
                    chk->chunk_counter--;
                    chk->last_op = H5SC_TAG_UNPIN_READ_DONE_ERROR;
                }
            }
        }
    }

    /* Restore base sel_chunks pointer for the caller (invoke batching expects this) */
    if (sc_io_info && base)
        sc_io_info->sel_chunks = base;

    if (addr) {
        for (size_t j = 0; j < chunk_count; j++)
            addr[j] = (haddr_t *)H5MM_xfree(addr[j]);
    }

    if (size) {
        for (size_t j = 0; j < chunk_count; j++)
            size[j] = (hsize_t *)H5MM_xfree(size[j]);
    }

    if (defined_values_size) {
        for (size_t j = 0; j < chunk_count; j++)
            defined_values_size[j] = (hsize_t *)H5MM_xfree(defined_values_size[j]);
    }

    if (size_hint) {
        for (size_t j = 0; j < chunk_count; j++)
            size_hint[j] = (size_t *)H5MM_xfree(size_hint[j]);
    }

    if (defined_values_size_hint) {
        for (size_t j = 0; j < chunk_count; j++)
            defined_values_size_hint[j] = (size_t *)H5MM_xfree(defined_values_size_hint[j]);
    }

    scaled                   = (const hsize_t **)H5MM_xfree((void *)scaled);
    addr                     = (haddr_t **)H5MM_xfree(addr);
    size                     = (hsize_t **)H5MM_xfree(size);
    defined_values_size      = (hsize_t **)H5MM_xfree(defined_values_size);
    size_hint                = (size_t **)H5MM_xfree(size_hint);
    defined_values_size_hint = (size_t **)H5MM_xfree(defined_values_size_hint);
    udata_arr                = (void **)H5MM_xfree(udata_arr);
    miss_arr                 = (size_t *)H5MM_xfree(miss_arr);
    chunk_arr                = (void **)H5MM_xfree(chunk_arr);

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
 *     caller's memory selection into chunk buffers, writes encoded chunk
 *     data to disk, updates SCC chunk state, and releases the
 *     per-selection pins established during I/O setup.
 *
 *   For each selected chunk in the active window, the function uses
 *     cached lookup metadata to distinguish resident cache hits from
 *     misses, creates new resident chunks for chunks not present on disk,
 *     reads and decodes on-disk chunks when necessary, invokes the layout
 *     client's gather_mem callback to update chunk contents from the user
 *     buffer, encodes the resulting chunk image, updates on-disk chunk
 *     indexing, and writes the encoded chunk to file.
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
 *   size_t count:
 *     Number of dataset I/O entries in dset_info. The current SCC write
 *     path expects count == 1. (TO BE REMOVED)
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
H5SC_write(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info)
{

    herr_t               ret_value = SUCCEED;
    size_t               alloc_size_total;
    H5D_io_type_info_t   my_io_type_info; /* Used in gather_mem callback */
    const H5S_t         *gather_mem_space;
    const H5S_t         *gather_file_space;
    haddr_t              md_tag                                  = HADDR_UNDEF;
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
    hsize_t             *old_disk_size                           = NULL;
    hsize_t            **defined_values_size                     = NULL;
    size_t             **size_hint                               = NULL;
    size_t             **defined_values_size_hint                = NULL;
    void               **udata_arr                               = NULL;
    hsize_t             *write_size_arr                          = NULL;
    void               **write_buf_arr                           = NULL;
    hsize_t             *write_buf_alloc_arr                     = NULL;
    size_t              *miss_arr                                = NULL;
    size_t              *hit_arr                                 = NULL;
    void               **chunk_arr                               = NULL;
    const hsize_t      **scaled_hit                              = NULL;
    haddr_t            **addr_hit                                = NULL;
    hsize_t            **size_hit                                = NULL;
    hsize_t            **defined_values_size_hit                 = NULL;
    size_t             **size_hint_hit                           = NULL;
    size_t             **defined_values_size_hint_hit            = NULL;
    void               **udata_hit                               = NULL;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
    assert(count == 0 || dset_info);
    assert(count == 1);

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
    assert(dset_info[0].dset->shared->layout.sc_ops->encode);
    assert(dset_info[0].dset->shared->layout.sc_ops->insert);

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

    /* Operate only on the chunks in the active window */
    chunk_count = sel_count;

    /* Allocate per-call scratch storage. */
    if (NULL == (scaled = (const hsize_t **)H5MM_malloc(chunk_count * sizeof(*scaled))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate scaled array");

    if (NULL == (addr = (haddr_t **)H5MM_malloc(chunk_count * sizeof(*addr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate addr array");

    if (NULL == (size = (hsize_t **)H5MM_malloc(chunk_count * sizeof(*size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate size array");

    if (NULL == (old_disk_size = (hsize_t *)H5MM_calloc(chunk_count * sizeof(*old_disk_size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate old_disk_size array");

    if (NULL == (defined_values_size = (hsize_t **)H5MM_malloc(chunk_count * sizeof(*defined_values_size))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate defined_values_size array");

    if (NULL == (size_hint = (size_t **)H5MM_malloc(chunk_count * sizeof(*size_hint))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate size_hint array");

    if (NULL ==
        (defined_values_size_hint = (size_t **)H5MM_malloc(chunk_count * sizeof(*defined_values_size_hint))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate defined_values_size_hint array");

    if (NULL == (udata_arr = (void **)H5MM_malloc(chunk_count * sizeof(*udata_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate udata_arr array");

    if (NULL == (write_size_arr = (hsize_t *)H5MM_calloc(chunk_count * sizeof(*write_size_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate write_size_arr array");

    if (NULL == (write_buf_arr = (void **)H5MM_calloc(chunk_count * sizeof(*write_buf_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate write_buf_arr array");

    if (NULL == (write_buf_alloc_arr = (hsize_t *)H5MM_calloc(chunk_count * sizeof(*write_buf_alloc_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate write_buf_alloc_arr array");

    if (NULL == (miss_arr = (size_t *)H5MM_malloc(chunk_count * sizeof(*miss_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate miss_arr array");

    if (NULL == (hit_arr = (size_t *)H5MM_malloc(chunk_count * sizeof(*hit_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate hit_arr array");

    if (NULL == (chunk_arr = (void **)H5MM_calloc(chunk_count * sizeof(*chunk_arr))))
        HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate chunk_arr array");

        /* Window-relevant selection accessor */
#define SEL(j_) (&base[sel_start + (j_)])

    /* Initialize per-chunk scratch state. */
    for (size_t j = 0; j < chunk_count; j++) {
        haddr_t *tmp_addr              = NULL;
        hsize_t *tmp_size              = NULL;
        hsize_t *tmp_def_val_size      = NULL;
        size_t  *tmp_size_hint         = NULL;
        size_t  *tmp_def_val_size_hint = NULL;

        scaled[j] = SEL(j)->scaled;

        tmp_addr = (haddr_t *)H5MM_malloc(sizeof(haddr_t));
        if (!tmp_addr)
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate chunk address scratch");
        *tmp_addr = HADDR_UNDEF;
        addr[j]   = tmp_addr;

        tmp_size = (hsize_t *)H5MM_malloc(sizeof(hsize_t));
        if (!tmp_size)
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate chunk size scratch");
        *tmp_size = 0;
        size[j]   = tmp_size;

        tmp_def_val_size = (hsize_t *)H5MM_malloc(sizeof(hsize_t));
        if (!tmp_def_val_size)
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate defined_values_size scratch");
        *tmp_def_val_size      = 0;
        defined_values_size[j] = tmp_def_val_size;

        tmp_size_hint = (size_t *)H5MM_malloc(sizeof(size_t));
        if (!tmp_size_hint)
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate size_hint scratch");
        *tmp_size_hint = 0;
        size_hint[j]   = tmp_size_hint;

        tmp_def_val_size_hint = (size_t *)H5MM_malloc(sizeof(size_t));
        if (!tmp_def_val_size_hint)
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                        "unable to allocate defined_values_size_hint scratch");
        *tmp_def_val_size_hint      = 0;
        defined_values_size_hint[j] = tmp_def_val_size_hint;

        udata_arr[j] = NULL;
    }

    /* Partition the active window into resident hits and lookup misses. */
    size_t miss_count = 0;
    size_t hit_count  = 0;

    if (H5SC__lookup_cache_misses(dset_info[0].dset, sc_io_info) < 0) {
        HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to lookup cache missed chunks (SCC)");
    }

    for (size_t j = 0; j < chunk_count; j++) {
        H5SC_chunk_t *chk = SEL(j)->cached_chunk;

        if (chk && chk->chunk_obj) {
            /* Cache hit: skip lookup; reuse persistent udata and on-disk identity */
            *addr[j]                     = chk->disk_addr;
            *size[j]                     = (hsize_t)chk->disk_nbytes;
            *size_hint[j]                = (size_t)chk->disk_nbytes;
            *defined_values_size[j]      = 0;
            *defined_values_size_hint[j] = 0;

            hit_arr[hit_count++] = j;
        }
        else {
            miss_arr[miss_count++] = j;
        }
    }

    /* Refresh per-call udata for resident cache hits before insert/writeback. */
    for (size_t j = 0; j < miss_count; j++) {
        size_t               idx = miss_arr[j];
        H5SC_io_sel_chunk_t *sel = SEL(idx);

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

    /* Refresh per-call udata for resident cache hits before insert/writeback. */
    if (hit_count > 0) {
        if (NULL == (scaled_hit = (const hsize_t **)H5MM_malloc(hit_count * sizeof(*scaled_hit))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate scaled_hit array");

        if (NULL == (addr_hit = (haddr_t **)H5MM_malloc(hit_count * sizeof(*addr_hit))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate addr_hit array");

        if (NULL == (size_hit = (hsize_t **)H5MM_malloc(hit_count * sizeof(*size_hit))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate size_hit array");

        if (NULL ==
            (defined_values_size_hit = (hsize_t **)H5MM_malloc(hit_count * sizeof(*defined_values_size_hit))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate defined_values_size_hit array");

        if (NULL == (size_hint_hit = (size_t **)H5MM_malloc(hit_count * sizeof(*size_hint_hit))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate size_hint_hit array");

        if (NULL == (defined_values_size_hint_hit =
                         (size_t **)H5MM_malloc(hit_count * sizeof(*defined_values_size_hint_hit))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL,
                        "unable to allocate defined_values_size_hint_hit array");

        if (NULL == (udata_hit = (void **)H5MM_calloc(hit_count * sizeof(*udata_hit))))
            HGOTO_ERROR(H5E_RESOURCE, H5E_NOSPACE, FAIL, "unable to allocate udata_hit array");

        for (size_t j = 0; j < hit_count; j++) {
            size_t idx                      = hit_arr[j];
            scaled_hit[j]                   = scaled[idx];
            addr_hit[j]                     = addr[idx];
            size_hit[j]                     = size[idx];
            defined_values_size_hit[j]      = defined_values_size[idx];
            size_hint_hit[j]                = size_hint[idx];
            defined_values_size_hint_hit[j] = defined_values_size_hint[idx];
        }

        if (dset_info[0].dset->shared->layout.sc_ops->lookup(
                dset_info[0].dset, hit_count, scaled_hit, addr_hit, size_hit, defined_values_size_hit,
                size_hint_hit, defined_values_size_hint_hit, udata_hit) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to refresh udata for cache hits (SCC)");

        for (size_t j = 0; j < hit_count; j++) {
            size_t idx     = hit_arr[j];
            udata_arr[idx] = udata_hit[j]; /* per-call only: do not persist into cached_chunk->udata */
        }
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
        H5SC_chunk_t *chk           = SEL(j)->cached_chunk;
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
                if (!(SEL(j)->cached_chunk && SEL(j)->cached_chunk->udata == udata_arr[j])) {
                    udata_arr[j] = H5MM_xfree(udata_arr[j]);
                }

                if (!size_hint[j] || *size_hint[j] == 0) {
                    HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL,
                                "write: invalid size_hint for new chunk allocation");
                }

                if (dset_info[0].dset->shared->layout.sc_ops->new_chunk(
                        dset_info[0].dset, false, size[j], size_hint[j], &chunk_arr[j], &udata_arr[j]) < 0) {
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to create new chunk (SCC)");
                }

                chk->chunk_obj   = chunk_arr[j];
                chk->disk_addr   = HADDR_UNDEF;
                chk->disk_nbytes = 0;
                chk->dirty_flag  = true;

                {
                    size_t resident_bytes = *size_hint[j];
                    if (resident_bytes == 0)
                        resident_bytes = (size_t)dset_info[0].dset->shared->layout.u.struct_chunk.size;

                    if (H5SC__chunk_update_cached_size(dset_hdr, chk, resident_bytes) < 0)
                        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                                    "write: chunk size accounting failed for new resident chunk");
                }
            }
        }
        else {
            old_disk_size[j] = *size[j];
            chk->disk_addr   = *addr[j];
            chk->disk_nbytes = (size_t)*size[j];

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
                if (dset_info[0].dset->shared->layout.sc_ops->decode(dset_info[0].dset, size[j], size_hint[j],
                                                                     partial_bound, &chunk_arr[j],
                                                                     udata_arr[j]) < 0) {
                    HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to decode chunk in place (SCC)");
                }
                /* After decode, chunk_arr[j] is in the in-cache format */
                chk->chunk_obj  = chunk_arr[j];
                chk->dirty_flag = false; /* (will become dirty after gather_mem updates) */

                /* Charge resident bytes for this chunk */
                size_t resident_bytes = *size_hint[j];
                if (resident_bytes == 0) {
                    resident_bytes = (size_t)dset_info[0].dset->shared->layout.u.struct_chunk.size;
                }

                if (H5SC__chunk_update_cached_size(dset_hdr, chk, resident_bytes) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL,
                                "write: chunk size accounting failed for decoded resident chunk");
            }
        }
    }

    for (size_t j = 0; j < chunk_count; j++) {

        /* Gather user data into the resident chunk buffer. */
        gather_mem_space  = SEL(j)->mem_space;
        gather_file_space = SEL(j)->file_space;

        if (dset_info[0].dset->shared->layout.sc_ops->gather_mem(
                &dset_info[0], &my_io_type_info, gather_mem_space, gather_file_space, size[j], size_hint[j],
                &alloc_size_total, chunk_arr[j], udata_arr[j]) < 0) {
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to gather mem for new chunk (SCC)");
        }

        /* After gather_mem, chunk is modified: mark resident chunk dirty */
        SEL(j)->cached_chunk->dirty_flag = true;
    }

    /* Encode resident chunks for writeback. */
    for (int j = 0; j < chunk_count; j++) {
        bool partial_bound = false;

        if (partial_bound_chunks_different_encoding && filtered &&
            H5D__chunk_is_partial_edge_chunk(dset_info[0].dset->shared->ndims,
                                             dset_info[0].dset->shared->layout.u.struct_chunk.dim, scaled[j],
                                             dset_info[0].dset->shared->curr_dims))
            partial_bound = true;

        /* Preserve cached decoded chunk objects: use encode (non-in-place) */
        if (dset_info[0].dset->shared->layout.sc_ops->encode(
                dset_info[0].dset, &write_size_arr[j], &write_buf_alloc_arr[j], partial_bound, chunk_arr[j],
                udata_arr[j], &write_buf_arr[j]) < 0) {
            HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "unable to encode chunk in place (SCC)");
        }
    }

    /* Insert the chunk into the chunk index within the file */
    if (dset_info[0].dset->shared->layout.sc_ops->insert(dset_info[0].dset, chunk_count, scaled, addr,
                                                         old_disk_size, write_size_arr, chunk_arr,
                                                         udata_arr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINSERT, FAIL, "unable to insert chunk into file (SCC)");

    for (size_t j = 0; j < chunk_count; j++) {

        /* Write the encoded chunk image to file. */
        if (H5F_block_write(dset_info[0].dset->oloc.file, H5FD_MEM_DRAW, *addr[j], (size_t)write_size_arr[j],
                            write_buf_arr[j]) < 0) {
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "unable to block write to file (SCC)");
        }

        /* insert() may update addr; reflect the final on-disk identity */
        {
            H5SC_chunk_t *chk = SEL(j)->cached_chunk;
            assert(chk);
            chk->disk_addr   = *addr[j];
            chk->disk_nbytes = (size_t)write_size_arr[j];
            chk->dirty_flag  = false;

            /*
             * Unpin: this chunk is no longer "in flight" for this write call.
             */
            if (chk->chunk_counter > 0) {
                chk->chunk_counter--;
                chk->last_op = H5SC_TAG_UNPIN_WRITE_DONE;
            }
            else {
                HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                            "SCC write: chunk counter was not properly incrimented");
            }
        }

        if (write_buf_arr[j]) {
            write_buf_arr[j] = H5MM_xfree(write_buf_arr[j]);
        }
        if (!(SEL(j)->cached_chunk && SEL(j)->cached_chunk->udata == udata_arr[j])) {
            udata_arr[j] = H5MM_xfree(udata_arr[j]);
        }
    }
    /* Reconcile dataset/global cache size accounting once per request */
    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "global size accounting failed (SCC write)");
    }

    H5AC_tag(md_tag, NULL); /* Reset the metadata tag for the next dataset */

    /* Close any temporary per-call dataspaces for the active window. */
    for (size_t j = 0; j < chunk_count; j++) {
        H5SC_io_sel_chunk_t *chk = SEL(j);

        if (chk->file_space && !chk->file_space_shared)
            if (H5S_close(chk->file_space) < 0)
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL,
                            "can't release temporary file dataspace (SCC write)");

        if (chk->mem_space && !chk->mem_space_shared)
            if (H5S_close(chk->mem_space) < 0)
                HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL,
                            "can't release temporary memory dataspace (SCC write)");

        /* Poison only the ephemeral dataspace pointers; leave lookup cache intact */
        chk->file_space        = NULL;
        chk->mem_space         = NULL;
        chk->file_space_shared = false;
        chk->mem_space_shared  = false;
        chk->dset_info         = NULL;
    }
#undef SEL

done:
    /*
     * Defensive unpin on error paths: if we abort mid-write, make sure any
     * chunks referenced by the current selection window are no longer marked
     * in-flight.
     */
    if (ret_value < 0) {
        if (sc_io_info && base && sel_count > 0) {
            for (size_t j = 0; j < sel_count; j++) {
                H5SC_chunk_t *chk = base[sel_start + j].cached_chunk;

                if (chk && chk->chunk_counter > 0) {
                    chk->chunk_counter--;
                    chk->last_op = H5SC_TAG_UNPIN_WRITE_DONE_ERROR;
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

    if (write_buf_arr) {
        for (size_t j = 0; j < chunk_count; j++)
            if (write_buf_arr[j])
                write_buf_arr[j] = H5MM_xfree(write_buf_arr[j]);
    }

    if (addr) {
        for (size_t j = 0; j < chunk_count; j++)
            addr[j] = (haddr_t *)H5MM_xfree(addr[j]);
    }

    if (size) {
        for (size_t j = 0; j < chunk_count; j++)
            size[j] = (hsize_t *)H5MM_xfree(size[j]);
    }

    if (defined_values_size) {
        for (size_t j = 0; j < chunk_count; j++)
            defined_values_size[j] = (hsize_t *)H5MM_xfree(defined_values_size[j]);
    }

    if (size_hint) {
        for (size_t j = 0; j < chunk_count; j++)
            size_hint[j] = (size_t *)H5MM_xfree(size_hint[j]);
    }

    if (defined_values_size_hint) {
        for (size_t j = 0; j < chunk_count; j++)
            defined_values_size_hint[j] = (size_t *)H5MM_xfree(defined_values_size_hint[j]);
    }

    scaled                   = (const hsize_t **)H5MM_xfree((void *)scaled);
    addr                     = (haddr_t **)H5MM_xfree(addr);
    size                     = (hsize_t **)H5MM_xfree(size);
    old_disk_size            = (hsize_t *)H5MM_xfree(old_disk_size);
    defined_values_size      = (hsize_t **)H5MM_xfree(defined_values_size);
    size_hint                = (size_t **)H5MM_xfree(size_hint);
    defined_values_size_hint = (size_t **)H5MM_xfree(defined_values_size_hint);
    udata_arr                = (void **)H5MM_xfree(udata_arr);
    write_size_arr           = (hsize_t *)H5MM_xfree(write_size_arr);
    write_buf_arr            = (void **)H5MM_xfree(write_buf_arr);
    write_buf_alloc_arr      = (hsize_t *)H5MM_xfree(write_buf_alloc_arr);
    miss_arr                 = (size_t *)H5MM_xfree(miss_arr);
    hit_arr                  = (size_t *)H5MM_xfree(hit_arr);
    chunk_arr                = (void **)H5MM_xfree(chunk_arr);

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
    assert(dset);
    assert(dset->shared->layout.sc_ops);
    assert(offset);
    assert(buf);
    assert(buf_size);

    HGOTO_ERROR(H5E_SCC, H5E_NOT_IMPLEMTED, FAIL, "the expected chunk was not found eligible for eviction");

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
    assert(dset);
    assert(dset->shared->layout.sc_ops);
    assert(offset);
    assert(buf);

    HGOTO_ERROR(H5E_SCC, H5E_NOT_IMPLEMTED, FAIL, "the expected chunk was not found eligible for eviction");

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC_direct_chunk_write() */

/*-------------------------------------------------------------------------
 * Function: H5SC_get_defined
 *
 * Purpose:  Returns a copy of file_space with only elements
 * selected that are both selected in file_space and defined
 * in dset. If file_space uses a point selection, the
 * ordering of selected points will be preserved in the
 * returned dataspace.
 *
 * Return:   SUCCEED on success, FAIL on failure
 *-------------------------------------------------------------------------
 */
H5S_t *
H5SC_get_defined(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space)
{
    H5S_t *defined   = NULL;
    H5S_t *ret_value = NULL;

    FUNC_ENTER_NOAPI(NULL)

    assert(cache);
    assert(dset);
    assert(dset->shared->layout.sc_ops);
    assert(file_space);

    /* FOR NOW: just return copy of file_space */
    if (NULL == (defined = H5S_copy(file_space, false, true)))
        HGOTO_ERROR(H5E_DATASET, H5E_CANTINIT, NULL, "unable to copy dataspace");

    /* Set return value */
    ret_value = defined;
    defined   = NULL;

done:
    if (defined) {
        assert(!ret_value);
        if (H5S_close(defined) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CLOSEERROR, NULL, "unable to release dataspace");
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
 *              Chunks that become empty are removed from structured storage,
 *              detached from the dataset LRU, removed from the SCC chunk hash
 *              table, and released. Chunks that retain defined values are
 *              marked dirty and flushed through the normal single-chunk SCC
 *              flush path.
 *
 *              The routine also preserves SCC pin/unpin accounting for chunks
 *              selected during erase initialization and reconciles dataset
 *              cache-size accounting after the erase operation completes.
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
    herr_t               ret_value     = SUCCEED;

    FUNC_ENTER_NOAPI(FAIL)

    assert(cache);
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

    if (sel_count == 0)
        HGOTO_DONE(SUCCEED);

    H5AC_tag(dset->oloc.addr, &md_tag);

#define SEL(j_) (&base[sel_start + (j_)])

    for (size_t j = 0; j < sel_count; j++) {
        H5SC_io_sel_chunk_t *sel          = SEL(j);
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
            hsize_t          def_local                               = 0;
            size_t           hint_local                              = 0;
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
                if (chk->chunk_counter > 0) {
                    chk->chunk_counter--;
                    chk->last_op = H5SC_TAG_UNPIN_ERASE_DONE;
                }
                else
                    HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                "SCC erase: chunk counter was not properly incremented");

                continue;
            }

            if (dset->shared->layout.sc_ops->lookup(dset, 1, scaled_arr, addr_arr, size_arr, def_sz_arr,
                                                    size_hint_arr, def_sz_hint_arr, udata_arr) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "erase: lookup callback failed");

            /* Sparse miss: no chunk exists, so nothing is defined here to erase */
            if (!H5_addr_defined(addr_local)) {
                if (udata_arr[0])
                    udata_arr[0] = H5MM_xfree(udata_arr[0]);

                if (chk->chunk_counter > 0) {
                    chk->chunk_counter--;
                    chk->last_op = H5SC_TAG_UNPIN_ERASE_DONE;
                }
                else
                    HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                "SCC erase: chunk counter was not properly incremented");

                continue;
            }

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

            if (dset->shared->layout.sc_ops->decode(dset, &size_local, &hint_local, partial_bound, &chunk_buf,
                                                    udata_arr[0]) < 0)
                HGOTO_ERROR(H5E_DATASET, H5E_CANTGET, FAIL, "erase: decode callback failed");

            chk->chunk_obj   = chunk_buf;
            chk->udata       = udata_arr[0];
            chk->disk_addr   = addr_local;
            chk->disk_nbytes = (size_t)size_local;
            chk->dirty_flag  = false;

            if (H5SC__chunk_update_cached_size(dset_hdr, chk, hint_local) < 0)
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

                if (chk->chunk_counter > 0) {
                    chk->chunk_counter--;
                    chk->last_op = H5SC_TAG_UNPIN_ERASE_DONE;
                }
                else
                    HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                                "SCC erase: chunk counter was not properly incremented");

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

            if (H5SC__chunk_lru_remove(dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL, "failed to remove chunk from dataset LRU");

            if (H5SC__chunk_teardown_resident(dset, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTRELEASE, FAIL,
                            "failed to tear down resident chunk state during erase");

            if (H5SC__ht_chunk_delete(cache, &chk->data_key) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL,
                            "failed to remove chunk from hash table during erase");

            chk = H5MM_xfree(chk);
            continue;
        }

        /* Surviving chunk: rewrite using the established single-chunk flush path */
        chk->dirty_flag = true;

        if (H5SC__flush_one_chunk(cache, dset, dset_hdr, chk) < 0)
            HGOTO_ERROR(H5E_DATASET, H5E_WRITEERROR, FAIL, "erase: unable to flush modified chunk");

        if (chk->chunk_counter > 0) {
            chk->chunk_counter--;
            chk->last_op = H5SC_TAG_UNPIN_ERASE_DONE;
        }
        else
            HGOTO_ERROR(H5E_SCC, H5E_BADOPCODE, FAIL,
                        "SCC erase: chunk counter was not properly incremented");
    }

    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0)
        HGOTO_ERROR(H5E_SCC, H5E_BADVALUE, FAIL, "global size accounting failed (SCC erase)");

    H5AC_tag(md_tag, NULL);

done:
    if (ret_value < 0) {
        if (sc_io_info && base && sel_count > 0) {
            for (size_t j = 0; j < sel_count; j++) {
                H5SC_chunk_t *chk = base[sel_start + j].cached_chunk;

                if (chk && chk->chunk_counter > 0) {
                    chk->chunk_counter--;
                    chk->last_op = H5SC_TAG_UNPIN_ERASE_DONE_ERROR;
                }
            }
        }
    }

    if (sc_io_info) {
        if (H5SC__io_info_reset(sc_io_info) < 0)
            HDONE_ERROR(H5E_DATASET, H5E_CANTRELEASE, FAIL, "can't close erase I/O info");
    }

#undef SEL
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
        H5S_t   *erase_space = NULL;
        hsize_t  start[H5S_MAX_RANK];
        hsize_t  count[H5S_MAX_RANK];
        hssize_t npoints = 0;
        bool     empty   = false;

        if (new_valid_count[u] >= old_valid_count[u])
            continue;

        if (NULL == (erase_space = H5S_create_simple((int)dset->shared->ndims, chunk_dims, NULL)))
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
                HDONE_ERROR(H5E_DATASPACE, H5E_CLOSEERROR, FAIL,
                            "unable to close erase dataspace after selection failure");
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTSELECT, FAIL,
                        "unable to select invalid chunk-local erase slab");
        }

        npoints = H5S_GET_SELECT_NPOINTS(erase_space);
        if (npoints < 0)
            HGOTO_ERROR(H5E_DATASPACE, H5E_CANTGET, FAIL,
                        "unable to get invalid chunk-local erase selection point count");

        if (npoints > 0) {
            if ((dset->shared->layout.sc_ops->erase_values)(dset, erase_space, nbytes, alloc_size,
                                                            chk->chunk_obj, delete_chunk, chk->udata) < 0) {
                if (H5S_close(erase_space) < 0)
                    HDONE_ERROR(H5E_DATASPACE, H5E_CLOSEERROR, FAIL,
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
    void          *udata = NULL;
    void         **udata_arr[1];
    herr_t         ret_value = SUCCEED;

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
    udata_arr[0]                    = &udata;

    if ((dset->shared->layout.sc_ops->lookup)(dset, (size_t)1, scaled_arr, addr_arr, size_arr,
                                              defined_values_size_arr, size_hint_arr,
                                              defined_values_size_hint_arr, udata_arr) < 0)
        HGOTO_ERROR(H5E_DATASET, H5E_NOTFOUND, FAIL, "unable to look up structured chunk for extent prune");

    if (H5_addr_defined(*addr))
        *exists = true;

done:
    if (udata)
        udata = H5MM_xfree(udata);

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

            if (H5SC__chunk_lru_remove(dset_hdr, chk) < 0)
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
                if (H5SC__chunk_update_cached_size(dset_hdr, chk, alloc_size) < 0)
                    HGOTO_ERROR(H5E_SCC, H5E_CANTSET, FAIL,
                                "unable to update SCC chunk cached size after extent prune erase");

                chk->disk_nbytes = nbytes;
                chk->dirty_flag  = true;
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

            if (H5SC__chunk_lru_remove(dset_hdr, chk) < 0)
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
            chk->dirty_flag = true;

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
    dset_hdr->evict_exhausted    = false;

    dset_hdr->io_info = (H5SC_io_info_t *)H5MM_calloc(sizeof(H5SC_io_info_t));
    if (!dset_hdr->io_info)
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, NULL, "can't allocate dataset I/O info");

    /* Default values: “empty but reusable” */
    dset_hdr->io_info->sel_chunks         = NULL;
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
    assert(dset);
    assert(dset_hdr);
    assert(dset_hdr->io_info);
    assert(dset_hdr->io_info->num_sel_chunks == 0);

    /* To avoid data loss: if the dataset still has resident chunks, evict them now */
    if (dset_hdr->lru_tail_ptr) {
        H5SC_chunk_t *chk = dset_hdr->lru_tail_ptr;
        while (chk) {
            H5SC_chunk_t *prev = chk->prev_ptr;

#ifdef H5SC_DO_SANITY_CHECKS
            assert(chk->chunk_counter == 0);
#endif

            if (H5SC__chunk_teardown_resident(dset, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANTRELEASE, FAIL,
                            "unable to tear down resident chunk object during dataset destroy");

            if (H5SC__chunk_lru_remove(dset_hdr, chk) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL,
                            "failed to remove chunk from dataset LRU during destroy");
            if (H5SC__ht_chunk_delete(cache, &chk->data_key) < 0)
                HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL,
                            "failed to remove chunk from hash table during destroy");

            chk = H5MM_xfree(chk);
            chk = prev;
        }
    }

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

done:
    if (!chk)
        H5MM_free(chk);

    return chk;
} /* end H5SC__make_chunk() */

/*-------------------------------------------------------------------------
 * Function: H5SC_make_and_insert_chunk
 *
 * Purpose:
 *   Helper that ensures a chunk corresponding to a given element coordinate
 *   is present in the shared chunk cache and attached to a dataset's chunk
 *   LRU list.
 *
 *   The function performs the following steps:
 *
 *     1) Computes the linear logical chunk index for the given element
 *        coordinate using H5SC__compute_logical_chunk_index().
 *
 *     2) Derives a 128-bit chunk key by interleaving the linear index and
 *        the dataset object header address via H5SC__compute_chunk_key().
 *        (Key convention: addr = linear chunk index; size = dset_addr.)
 *
 *     3) Queries the chunk hash table for an existing entry matching the
 *        computed key. If an entry exists:
 *           - *out_chunk is set to the found chunk
 *           - *out_key is set to the computed key
 *           - SUCCEED is returned without modifying the LRU.
 *
 *     4) If no entry exists in the hash table, allocates a new H5SC_chunk_t
 *        via t_make_chunk(), initializes its payload (including data_key),
 *        inserts it into the chunk hash table, and prepends it to the
 *        dataset's chunk LRU list.
 *
 *     5) Performs an identity check by re-looking up the key in the hash
 *        table and verifying that the returned pointer matches the inserted
 *        H5SC_chunk_t.
 *
 * Inputs:
 *   H5SC_t *cache:
 *     Pointer to the shared chunk cache instance (hash tables + LRU state).
 *
 *   H5SC_dset_header_t *dset_hdr:
 *     Pointer to the dataset header whose per-dataset chunk LRU should
 *     contain the chunk. Must remain valid for the lifetime of the chunk.
 *
 *   haddr_t daddr:
 *     Dataset object header address. Used as one half of the interleaved
 *     chunk key (paired with the linear chunk index).
 *
 *   unsigned ndims:
 *     Rank of the dataset (number of dimensions).
 *
 *   const hsize_t *dims:
 *     Array of length nd containing the dataset dimensions (in elements).
 *
 *   const uint32_t *cdims:
 *     Array of length nd containing the chunk dimensions (in elements).
 *
 *   const hsize_t *elem:
 *     Element coordinate (length nd) that lies inside the chunk for which
 *     we are ensuring cache residency. This coordinate is used to derive
 *     the logical chunk coordinates and the linear chunk index.
 *
 *   size_t cached_sz:
 *     Size (in bytes) to assign to both cached_chunk_size and disk_chunk_size
 *     in the newly created chunk (if one is allocated). This is used for
 *     accounting and LRU byte counters in the tests.
 *
 * Outputs:
 *   H5SC_chunk_t **out_chunk:
 *     On success, *out_chunk is set to point at the chunk that now represents
 *     this logical chunk in the cache. This may be an existing chunk (if one
 *     was already present in the hash table) or a newly allocated chunk.
 *
 *   H5SC_chunk_key_t *out_key:
 *     On success, *out_key is set to the computed chunk key corresponding
 *     to the given element coordinate and dataset address.
 *
 * Return:
 *   SUCCEED (0) on success, FAIL (<0) on error. On failure, no new chunk
 *   remains linked in the hash table or the LRU list, and no allocation
 *   is leaked.
 *------------------------------------------------------------------------- */
herr_t
H5SC_make_and_insert_chunk(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, haddr_t daddr, unsigned ndims,
                           const hsize_t *dims, hsize_t *cdims,
                           const hsize_t    *elem,      /* element coord inside the chunk */
                           size_t            cached_sz, /* cached (and disk) size for counters */
                           H5SC_chunk_t    **out_chunk, /* [out] chunk pointer (existing or new) */
                           H5SC_chunk_key_t *out_key /* [out] computed key */)
{
    herr_t ret_value = SUCCEED;

    hsize_t       idx = 0;
    H5SC_chunk_t *chk = NULL;
    FUNC_ENTER_NOAPI(FAIL)

    if (!cache || !dset_hdr || !dims || !cdims || !elem || !out_chunk || !out_key) {
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "invalid argument(s) (make and insert chunk, SCC)");
    }

    /* 1) Compute linear chunk index from element coordinate */
    if (H5SC__compute_logical_chunk_index(ndims, dims, cdims, elem, &idx) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CHUNK_KEY, FAIL, "failed to compute the logical chunk coordinate");
    }

    /* After computing idx */
#ifdef H5SC_DO_SANITY_CHECKS
    {
        /* Basic argument sanity (fast checks) */
        assert(ndims > 0);
        assert(dims);
        assert(cdims);
        assert(elem);
        for (unsigned i = 0; i < ndims; i++) {
            assert(dims[i] > 0);
            assert(cdims[i] > 0);
            assert(elem[i] < dims[i]);
        }

        /* idx must be within total chunk count (product of nchunks per dim) */
        {
            hsize_t total = 1;
            for (unsigned i = 0; i < ndims; i++) {
                hsize_t n = (dims[i] + cdims[i] - 1) / cdims[i];
                assert(n > 0);
                total *= n;
            }
            assert(idx < total);
        }
    }
#endif

/* After computing out_key */
#ifdef H5SC_DO_SANITY_CHECKS
    {
        /* Key should not be all-zero unless both components are zero */
        if (daddr != (haddr_t)0 || idx != (hsize_t)0)
            assert(!(out_key->high_half == 0 && out_key->low_half == 0));
    }
#endif

    /* 2) Compute chunk key: interleave (dset_addr, linear_idx) */
    if (H5SC__compute_chunk_key(&daddr, &idx, out_key) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CHUNK_KEY, FAIL, "failed to compute the chunk key coordinate");
    }

    /* Key sanity check (post-computation); keys should never be all-zero */
    if (out_key->high_half == 0 && out_key->low_half == 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CHUNK_KEY, FAIL, "computed chunk key should never be all-zero");
    }

    /* 3) Lookup existing chunk in hash table */
    chk = H5SC__ht_chunk_find(cache, out_key);
    if (chk) {
        /* Check that the cached chunk has the correct scaled coordinates (for callback usage) */
        for (unsigned u = 0; u < ndims; u++) {
            hsize_t size = elem[u] / cdims[u];
#ifdef H5SC_DO_SANITY_CHECKS
            assert(chk->scaled[u] == 0 || chk->scaled[u] == size);
#endif
            chk->scaled[u] = size;
        }

        /* Existing chunk: just hand it back and return success */
        *out_chunk = chk;

#ifdef H5SC_DO_SANITY_CHECKS
        assert(chk->magic == H5SC_CHUNK_MAGIC);
        assert(chk->data_key.high_half == out_key->high_half);
        assert(chk->data_key.low_half == out_key->low_half);
#endif

        return SUCCEED;
    }

    /* 4) If not found: ensure space, then allocate and insert new chunk */

    if (H5SC__ensure_space(cache, cached_sz) < 0) {
        HGOTO_ERROR(H5E_SCC, H5E_CANNOTEVICT, FAIL, "failed to make room for for new chunk (SCC)");
    }

    chk = H5SC__make_chunk(*out_key, cached_sz, 0, false);

    /* Fail if the chunk is NULL */
    if (!chk) {
        HGOTO_ERROR(H5E_RESOURCE, H5E_CANTALLOC, FAIL, "failed to allocate chunk (SCC)");
    }

    /* Persist scaled coordinates nedeed by the callbacks */
    for (unsigned u = 0; u < ndims; u++) {
        chk->scaled[u] = elem[u] / cdims[u];
    }

    /* Set the chunk tag to 1 after creation */

    if (H5SC__ht_chunk_insert(cache, chk) < 0) {
        H5MM_xfree(chk);
        return FAIL;
    }
    chk->last_op = H5SC_TAG_HT_INSERT;

    size_t old_dset_size = dset_hdr->curr_dset_size;

    if (H5SC__chunk_lru_prepend(dset_hdr, chk) < 0) {
        (void)H5SC__ht_chunk_delete(cache, out_key);
        H5MM_xfree(chk);
        return FAIL;
    }

    chk->last_op = H5SC_TAG_LRU_INSERT;

    /* Update dataset and cache size accounting */
    if (H5SC__account_chunk_link_change(cache, dset_hdr, old_dset_size) < 0) {
        chk->last_op = H5SC_TAG_EVICT;
        (void)H5SC__chunk_lru_remove(dset_hdr, chk);
        (void)H5SC__ht_chunk_delete(cache, out_key);
        H5MM_xfree(chk);
        return FAIL;
    }

    /* 5) Identity check: HT lookup must return the same pointer */
    {
        H5SC_chunk_t *got = H5SC__ht_chunk_find(cache, out_key);
        if (got != chk) {
            chk->last_op = H5SC_TAG_EVICT;
            (void)H5SC__chunk_lru_remove(dset_hdr, chk);
            (void)H5SC__ht_chunk_delete(cache, out_key);
            H5MM_xfree(chk);
            return FAIL;
        }
    }

    *out_chunk = chk;

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

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
 * Function: H5SC__set_interleave_bit
 *
 * Purpose:
 *   Helper function that sets a specified bit position within the
 *     128-bit chunk key (H5SC_chunk_key_t), which is represented as
 *     two uint64_t halves (low_half and high_half).
 *
 *   The function determines whether the target bit lies in the lower
 *     or upper 64-bit segment and sets the corresponding bit. It is
 *     intended to be used during construction of interleaved chunk
 *     keys.
 *
 * Inputs:
 *   H5SC_chunk_key_t *chunk_key:
 *     Pointer to the chunk key structure whose bit field is to be
 *     modified. The structure is assumed to be properly initialized
 *     prior to invocation.
 *
 *   unsigned bit_position:
 *     The bit position within the 128-bit key to set (range: 0–127).
 *
 * Return:
 *   None.
 *-------------------------------------------------------------------------
 */

static inline void
H5SC__set_interleave_bit(H5SC_chunk_key_t *chunk_key, unsigned bit_position)
{

    /* Prior to this function being called, all necessary
     * inputs should be checked and/or verified, allowing
     * this function to be light-weight.
     */

    if (bit_position < (unsigned)64) {
        chunk_key->low_half |= ((uint64_t)1 << bit_position);
    }
    else {
        chunk_key->high_half |= ((uint64_t)1 << (bit_position - (unsigned)64));
    }
} /* end H5SC__set_interleave_bit() */

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
H5SC__compute_chunk_key(haddr_t *dset_object_header_addr /*in*/, hsize_t *log_chk_coord /*in*/,
                        H5SC_chunk_key_t *chunk_key /*in,out*/)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(dset_object_header_addr);
    assert(log_chk_coord);
    assert(chunk_key);

    /* Normalize inputs to 64-bit values for bit interleaving. */
    uint64_t addr    = (uint64_t)(*dset_object_header_addr);
    uint64_t log_chk = (uint64_t)(*log_chk_coord);

    /* Clear the destination key before populating it. */
    chunk_key->high_half = (uint64_t)0;
    chunk_key->low_half  = (uint64_t)0;

    /* Interleave dataset address bits and logical chunk index bits. */
    for (unsigned i = 0; i < (unsigned)64; ++i) {
        if (((addr >> i) & (uint64_t)1) != 0)
            H5SC__set_interleave_bit(chunk_key, (unsigned)2 * i);

        if (((log_chk >> i) & (uint64_t)1) != 0)
            H5SC__set_interleave_bit(chunk_key, (unsigned)2 * i + (unsigned)1);
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
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
H5SC__compute_logical_chunk_index(unsigned ndims, const hsize_t *dset_dims, hsize_t *chunk_dims,
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

    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);

#ifdef H5SC_DO_SANITY_CHECKS
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

#ifdef H5SC_DO_SANITY_CHECKS
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

#ifdef H5SC_DO_SANITY_CHECKS
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

#ifdef H5SC_DO_SANITY_CHECKS
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
 * Function:    H5SC__dset_lru_append
 *
 * Purpose:
 *   Insert a dataset header as LRU (at the tail) of the global dataset-header
 *   LRU maintained in H5SC_t.{dset_lru_head_ptr,dset_lru_tail_ptr}. Pointer
 *   splicing is performed by H5SC_DLL_APPEND; counters are updated explicitly.
 *
 * Inputs:
 *   H5SC_t             *cache     - SCC instance holding global header LRU
 *   H5SC_dset_header_t *dset_hdr  - Header to insert (must not already be linked)
 *
 * Returns:
 *   SUCCEED on success; FAIL on already-linked, splice, or invariant error.
 */
herr_t
H5SC__dset_lru_append(H5SC_t *cache, H5SC_dset_header_t *dset_hdr)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *it        = NULL;
    H5SC_dset_header_t *pr        = NULL;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(dset_hdr);
    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);

#ifdef H5SC_DO_SANITY_CHECKS
    if (dset_hdr->next_dset_ptr || dset_hdr->prev_dset_ptr || cache->dset_lru_head_ptr == dset_hdr)
        HGOTO_ERROR(H5E_SCC, H5E_ALREADY_LINKED, FAIL, "header already linked in global LRU");
#endif

    H5SC_DLL_APPEND(dset_hdr, next_dset_ptr, prev_dset_ptr, cache->dset_lru_head_ptr,
                    cache->dset_lru_tail_ptr,
                    HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "header append failed"));

    cache->dset_lru_len++;
    cache->SCC_quiescent_size += dset_hdr->curr_dset_size;

    dset_hdr->last_op = H5SC_TAG_LRU_TOUCH;

#ifdef H5SC_DO_SANITY_CHECKS
    H5SC_DLL_CHECK_LINKS(it, pr, cache->dset_lru_head_ptr, cache->dset_lru_tail_ptr, next_dset_ptr,
                         prev_dset_ptr,
                         HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL, "bad header LRU linkage"));
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__dset_lru_append() */

/******************************************************************************
 *
 * Function:    H5SC__dset_exhausted_lru_prepend
 *
 * Purpose:
 *   Insert a dataset header into the optional exhausted
 * list (caller-maintained head/tail) as MRU. Uses
 * next_exhausted_ptr/prev_exhausted_ptr links.
 *
 * Inputs:
 *   H5SC_exhausted_list_t *exh – Exhausted list endpoints &
 * counters H5SC_dset_header_t    *dset_hdr – Header to
 * insert (must not already be linked)
 *
 * Returns
 *   SUCCEED on success; FAIL on already-linked, splice, or
 * invariant error.
 */

static herr_t
H5SC__dset_exhausted_lru_prepend(H5SC_exhausted_list_t *exh, H5SC_dset_header_t *dset_hdr)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *it = NULL, *pr = NULL;
    FUNC_ENTER_PACKAGE

#ifdef H5SC_DO_SANITY_CHECKS
    if (dset_hdr->next_exhausted_ptr || dset_hdr->prev_exhausted_ptr || exh->head == dset_hdr)
        HGOTO_ERROR(H5E_SCC, H5E_ALREADY_LINKED, FAIL,
                    "header already linked in "
                    "exhausted list");
#endif

    H5SC_DLL_PREPEND(dset_hdr, next_exhausted_ptr, prev_exhausted_ptr, exh->head, exh->tail,
                     HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "exhausted prepend failed"));

    exh->len++;
    exh->bytes += dset_hdr->curr_dset_size;

#ifdef H5SC_DO_SANITY_CHECKS
    H5SC_DLL_CHECK_LINKS(it, pr, exh->head, exh->tail, next_exhausted_ptr, prev_exhausted_ptr,
                         HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL, "bad exhausted linkage"));
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

/******************************************************************************
 *
 * Function:    H5SC__dset_exhausted_remove
 *
 * Purpose:
 *   Remove a dataset header from the optional exhausted
 * list and decrement the list counters by
 * dset_hdr->curr_dset_size.
 *
 * Inputs:
 *   H5SC_exhausted_list_t *exh – Exhausted list endpoints &
 * counters H5SC_dset_header_t    *dset_hdr – Header to
 * remove (must be linked)
 *
 * Returns
 *   SUCCEED on success; FAIL if the header is not a member
 * or splice fails.
 */

static herr_t
H5SC__dset_exhausted_remove(H5SC_exhausted_list_t *exh, H5SC_dset_header_t *dset_hdr)
{
    herr_t              ret_value = SUCCEED;
    H5SC_dset_header_t *it = NULL, *pr = NULL;
    FUNC_ENTER_PACKAGE

#ifdef H5SC_DO_SANITY_CHECKS
    if (!(dset_hdr->next_exhausted_ptr || dset_hdr->prev_exhausted_ptr || exh->head == dset_hdr))
        HGOTO_ERROR(H5E_SCC, H5E_NOT_A_MEMBER, FAIL, "header not in exhausted list");
#endif

    H5SC_DLL_REMOVE(dset_hdr, next_exhausted_ptr, prev_exhausted_ptr, exh->head, exh->tail,
                    HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "exhausted remove failed"));

    exh->len--;
    exh->bytes -= dset_hdr->curr_dset_size;

#ifdef H5SC_DO_SANITY_CHECKS
    H5SC_DLL_CHECK_LINKS(it, pr, exh->head, exh->tail, next_exhausted_ptr, prev_exhausted_ptr,
                         HGOTO_ERROR(H5E_SCC, H5E_DLL_INVARIANT, FAIL, "bad exhausted linkage"));
#endif

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

/*-------------------------------------------------------------------------
 * Function: H5SC__dset_exhausted_restore_to_global_lru
 *
 * Purpose:
 *   Function that restores dataset headers from the exhausted-dataset list
 *     back to the global dataset LRU. Headers are removed from the
 *     exhausted list one at a time from head to tail and appended to the
 *     tail of the global dataset LRU so that their original old-to-new
 *     traversal order is preserved for subsequent eviction passes.
 *
 * Inputs:
 *   H5SC_t *cache               - Pointer to the SCC cache state whose
 *      global dataset LRU is to be repopulated.
 *
 *   H5SC_exhausted_list_t *exh -  Pointer to the exhausted-dataset list
 *      containing headers that were fully processed during the current eviction pass.
 *
 * Return:
 *   SUCCEED on success;
 *   FAIL on failure.
 *-------------------------------------------------------------------------
 */

static herr_t
H5SC__dset_exhausted_restore_to_global_lru(H5SC_t *cache, H5SC_exhausted_list_t *exh)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    assert(cache);
    assert(exh);

    /* Restore in exhausted head->tail order so original global order is preserved at the tail. */
    while (exh->head) {
        H5SC_dset_header_t *dset_hdr = exh->head;

        if (H5SC__dset_exhausted_remove(exh, dset_hdr) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANNOTREMOVE, FAIL, "failed to remove dataset from exhausted list");

        if (H5SC__dset_lru_append(cache, dset_hdr) < 0)
            HGOTO_ERROR(H5E_SCC, H5E_CANTINSERT, FAIL, "failed to append dataset back to global LRU");
    }

done:
    FUNC_LEAVE_NOAPI(ret_value)
}

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
H5SC__chunk_lru_prepend(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk)
{
    herr_t        ret_value = SUCCEED;
    H5SC_chunk_t *it = NULL, *pr = NULL;

    FUNC_ENTER_PACKAGE

    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);
    assert(chunk->magic == H5SC_CHUNK_MAGIC);

#ifdef H5SC_DO_SANITY_CHECKS
    if (chunk->next_ptr || chunk->prev_ptr || dset_hdr->lru_head_ptr == chunk)
        HGOTO_ERROR(H5E_SCC, H5E_ALREADY_LINKED, FAIL, "chunk already linked in per-dataset LRU");
#endif

    /* Splice first; if it fails, nothing changes. */
    H5SC_DLL_PREPEND(chunk, next_ptr, prev_ptr, dset_hdr->lru_head_ptr, dset_hdr->lru_tail_ptr,
                     HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "chunk prepend failed"));

    /* Embedded counters (authoritative for the per-dataset LRU). */
    dset_hdr->chunk_lru_len++;
    dset_hdr->curr_dset_size += chunk->cached_chunk_size;

#ifdef H5SC_DO_SANITY_CHECKS
    assert(dset_hdr->curr_dset_size >= chunk->cached_chunk_size);
#endif

    /* Add the struct tags */
    dset_hdr->last_op = H5SC_TAG_LRU_TOUCH;
    chunk->last_op    = H5SC_TAG_LRU_PROMOTE;

#ifdef H5SC_DO_SANITY_CHECKS
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
H5SC__chunk_lru_remove(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk)
{
    herr_t        ret_value = SUCCEED;
    H5SC_chunk_t *it = NULL, *pr = NULL;

    FUNC_ENTER_PACKAGE

    if (!(chunk->next_ptr || chunk->prev_ptr || dset_hdr->lru_head_ptr == chunk))
        HGOTO_ERROR(H5E_SCC, H5E_NOT_A_MEMBER, FAIL, "chunk not in per-dataset LRU");

    assert(dset_hdr->magic == H5SC_DSET_HDR_MAGIC);
    assert(chunk->magic == H5SC_CHUNK_MAGIC);

    /* Splice out first; if it fails, leave counters untouched. */
    H5SC_DLL_REMOVE(chunk, next_ptr, prev_ptr, dset_hdr->lru_head_ptr, dset_hdr->lru_tail_ptr,
                    HGOTO_ERROR(H5E_SCC, H5E_SPLICE_FAILED, FAIL, "chunk remove failed"));

    chunk->last_op    = H5SC_TAG_LRU_REMOVE;
    dset_hdr->last_op = H5SC_TAG_LRU_REMOVE;

    /* Embedded counters */
    dset_hdr->chunk_lru_len--;

#ifdef H5SC_DO_SANITY_CHECKS
    if (chunk->cached_chunk_size > dset_hdr->curr_dset_size)
        HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "chunk bytes underflow");
#endif

    dset_hdr->curr_dset_size -= chunk->cached_chunk_size;

#ifdef H5SC_DO_SANITY_CHECKS
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
 *   Adjust counters to reflect a change in chunk->cached_chunk_size (no relink)
 * and commit the new size into the chunk object.
 *
 * Inputs:
 *   H5SC_dset_header_t dset_hdr – Dataset containing the target chunk
 *   H5SC_chunk_t *chunk         – Target chunk (must already exist in the cache)
 *   size_t new_size             – New cached size in bytes
 *
 * Returns
 *   SUCCEED on success; FAIL if size math would violate invariants.
 */

herr_t
H5SC__chunk_update_cached_size(H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk, size_t new_size)
{
    herr_t ret_value = SUCCEED;

    size_t old = chunk->cached_chunk_size;

    FUNC_ENTER_PACKAGE

    assert(dset_hdr);
    assert(chunk);

    if (new_size == old) {
        HGOTO_DONE(SUCCEED); /* no-op */
    }

    bool linked = (chunk->next_ptr || chunk->prev_ptr || dset_hdr->lru_head_ptr == chunk);

    if (linked) {

        if (new_size > old) {
            size_t delta = new_size - old;
            dset_hdr->curr_dset_size += delta;
        }
        else {
            size_t delta = old - new_size;
#ifdef H5SC_DO_SANITY_CHECKS
            if (delta > dset_hdr->curr_dset_size)
                HGOTO_ERROR(H5E_SCC, H5E_SIZE_MISMATCH, FAIL, "chunk bytes underflow");
#endif

            dset_hdr->curr_dset_size -= delta;
        }
    }

    chunk->cached_chunk_size = new_size;

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5SC__chunk_update_cached_size() */

/* -------------------------------------------------------------
 * Debug helpers
 * -------------------------------------------------------------
 */
#if defined(H5SC_ENABLE_DUMPS) && (H5SC_ENABLE_DUMPS + 0)
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
#endif /* H5SC_ENABLE_DUMPS */

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
        found->last_op = H5SC_TAG_LOOKUP;
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

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* H5SC_validate_config() */

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

    cache->stats.scc_evictions++;
} /* end H5SC__stats_record_eviction() */

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

    if (0 == cache->stats.scc_lookups)
        return 0.0;

    return ((double)cache->stats.scc_hits / (double)cache->stats.scc_lookups);
} /* end H5SC__stats_get_hit_rate() */

/******************************************************************************
 *
 * Function:    H5SC_stats_get_miss_rate
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
H5SC_stats_get_miss_rate(const H5SC_t *cache)
{
    assert(cache);

    if (0 == cache->stats.scc_lookups)
        return 0.0;

    return ((double)cache->stats.scc_misses / (double)cache->stats.scc_lookups);
} /* end H5SC_stats_get_miss_rate() */

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

herr_t
H5SC__stats_dump(const H5SC_t *cache)
{
    herr_t ret_value = SUCCEED;

    FUNC_ENTER_PACKAGE

    if (!cache)
        HGOTO_ERROR(H5E_ARGS, H5E_BADVALUE, FAIL, "invalid cache pointer");

    fprintf(stdout, "H5SC cache statistics:\n");
    fprintf(stdout, "    lookups:   %" PRIuHSIZE "\n", cache->stats.scc_lookups);
    fprintf(stdout, "    hits:      %" PRIuHSIZE "\n", cache->stats.scc_hits);
    fprintf(stdout, "    misses:    %" PRIuHSIZE "\n", cache->stats.scc_misses);
    fprintf(stdout, "    hit rate:  %.2f%%\n", 100.0 * H5SC__stats_get_hit_rate(cache));
    fprintf(stdout, "    miss rate: %.2f%%\n", 100.0 * H5SC_stats_get_miss_rate(cache));
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

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__stats_dump() */

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

done:
    FUNC_LEAVE_NOAPI(ret_value)
} /* end H5SC__get_cache_from_file_id() */