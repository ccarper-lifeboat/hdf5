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

/*-------------------------------------------------------------------------
 *
 * Created:     H5SCprivate.h
 *
 * Purpose:     Private header for library accessible shared chunk cache routines.
 *
 *-------------------------------------------------------------------------
 */

#ifndef H5SCprivate_H
#define H5SCprivate_H

/* Forward declarations that may be needed by the below headers */
typedef struct H5SC_layout_ops_t H5SC_layout_ops_t;

/* Private headers needed by this file */
#include "H5private.h"  /* Generic Functions                   */
#include "H5Dprivate.h" /* Datasets                            */
#include "H5SCpublic.h" /* Public prototypes                   */

/**************************/
/* Library Private Macros */
/**************************/

/*
 * Enable SCC invariant checking in assertion-enabled builds. Define this in
 * the private header because sanity-only private declarations may be consumed
 * without H5SCpkg.h having been included first.
 */
#ifndef H5SC_DO_SANITY_CHECKS
#ifndef NDEBUG
#define H5SC_DO_SANITY_CHECKS 1
#else
#define H5SC_DO_SANITY_CHECKS 0
#endif
#endif

#if !((H5SC_DO_SANITY_CHECKS == 0) || (H5SC_DO_SANITY_CHECKS == 1))
#error "The value of H5SC_DO_SANITY_CHECKS must be 0 or 1."
#endif

/* clang-format off */
#define H5SC__DEFAULT_SCC_CONFIG                  \
{                                                 \
    /* version    = */ H5SC__CURR_SCC_VERSION,    \
    /* max_q_size = */ ((size_t)(1000ULL * 1024ULL * 1024ULL)),   \
    /* max_a_size = */ ((size_t)(2ULL *1000ULL * 1024ULL * 1024ULL))  \
}
/* clang-format on */

#define H5SC_CHUNK_ADD(head_, item_)                                                                         \
    HASH_ADD_KEYPTR(hval_chunk, (head_), &(item_)->data_key, sizeof((item_)->data_key), (item_))
#define H5SC_CHUNK_FIND(head_, keyptr_, out_)                                                                \
    HASH_FIND(hval_chunk, (head_), (keyptr_), sizeof(*(keyptr_)), (out_))
#define H5SC_CHUNK_DEL(head_, item_)      HASH_DELETE(hval_chunk, (head_), (item_))
#define H5SC_CHUNK_ITER(head_, el_, tmp_) HASH_ITER(hval_chunk, (head_), (el_), (tmp_))

#define H5SC_DSET_ADD(head_, item_)                                                                          \
    HASH_ADD_KEYPTR(dset_hval, (head_), &(item_)->dset_addr, sizeof((item_)->dset_addr), (item_))
#define H5SC_DSET_FIND(head_, addrptr_, out_)                                                                \
    HASH_FIND(dset_hval, (head_), (addrptr_), sizeof(*(addrptr_)), (out_))
#define H5SC_DSET_DEL(head_, item_)      HASH_DELETE(dset_hval, (head_), (item_))
#define H5SC_DSET_ITER(head_, el_, tmp_) HASH_ITER(dset_hval, (head_), (el_), (tmp_))

/****************************/
/* Library Private Typedefs */
/****************************/

/* Forward declaration for shared chunk cache structs (defined in H5SCpkg.h) */

/******************************************************************************
 *
 * Enumeration: H5SC_tag_t
 *
 * Info
 *
 *      The enumeration defines tags used to identify the most recent internal
 *      SCC operation performed on an H5SC_chunk_t or H5SC_dset_header_t
 *      structure. These values are primarily used for debugging, validation,
 *      and tracking state transitions during hash table, LRU, I/O, flush, and
 *      eviction operations.
 *
 *      An H5SC_tag_t value is stored in the last_op field of both
 *      H5SC_chunk_t and H5SC_dset_header_t. The structure containing the tag
 *      provides the context necessary to interpret operations that are common
 *      to both structure types. For example, H5SC_TAG_LRU_INSERT stored in an
 *      H5SC_chunk_t refers to insertion into a dataset's chunk LRU list, while
 *      the same tag stored in an H5SC_dset_header_t refers to insertion into
 *      the SCC dataset LRU list.
 *
 *      Tags are therefore operation-oriented rather than structure-oriented.
 *      A separate tag is not required for equivalent operations performed on
 *      chunks and dataset headers. Some tags, particularly those associated
 *      with pinning and unpinning during I/O, apply only to H5SC_chunk_t
 *      structures.
 *
 * Values
 *
 *  H5SC_TAG_NONE – Indicates that no SCC operation has been associated
 *      with the structure. This value is used as an uninitialized or default
 *      tag where appropriate.
 *
 *  H5SC_TAG_CREATE – Indicates that the H5SC_chunk_t or
 *      H5SC_dset_header_t structure was created.
 *
 *  H5SC_TAG_HT_INSERT – Indicates that the structure was inserted into its
 *      associated SCC hash table. For H5SC_chunk_t, this refers to the chunk
 *      hash table. For H5SC_dset_header_t, this refers to the dataset hash
 *      table.
 *
 *  H5SC_TAG_HT_DELETE – Indicates that the structure was removed from its
 *      associated SCC hash table.
 *
 *  H5SC_TAG_LRU_INSERT – Indicates that the structure was inserted into its
 *      associated LRU list. For H5SC_chunk_t, this refers to the associated
 *      dataset's chunk LRU list. For H5SC_dset_header_t, this refers to the
 *      SCC dataset LRU list.
 *
 *  H5SC_TAG_LRU_REMOVE – Indicates that the structure was removed from its
 *      associated LRU list.
 *
 *  H5SC_TAG_LRU_PROMOTE – Indicates that the structure was promoted within
 *      its associated LRU list to reflect recent use.
 *
 *  H5SC_TAG_LRU_TOUCH – Indicates that the recency of the structure was
 *      updated as the result of an access or other operation affecting its
 *      position within an LRU list.
 *
 *  H5SC_TAG_FLUSH_DIRTY – Indicates that the structure was involved in an
 *      operation responsible for flushing dirty cached chunk data.
 *
 *  H5SC_TAG_FLUSH – Indicates that the structure was involved in a general
 *      SCC flush operation.
 *
 *  H5SC_TAG_EVICT – Indicates that the structure was involved in an SCC
 *      eviction operation.
 *
 *  H5SC_TAG_LOOKUP – Indicates that the structure was accessed during an
 *      SCC lookup operation.
 *
 *  H5SC_TAG_PIN_IOINIT – Indicates that a chunk was pinned while
 *      initializing an I/O request. This tag applies to H5SC_chunk_t
 *      structures.
 *
 *  H5SC_TAG_PIN_IOINIT_CACHED – Indicates that an existing cached chunk
 *      was pinned while initializing an I/O request. This tag applies to
 *      H5SC_chunk_t structures.
 *
 *  H5SC_TAG_UNPIN_WRITE_DONE – Indicates that a chunk was unpinned after
 *      successful completion of a write operation. This tag applies to
 *      H5SC_chunk_t structures.
 *
 *  H5SC_TAG_UNPIN_WRITE_DONE_ERROR – Indicates that a chunk was unpinned
 *      while handling an error during completion of a write operation. This
 *      tag applies to H5SC_chunk_t structures.
 *
 *  H5SC_TAG_UNPIN_READ_DONE – Indicates that a chunk was unpinned after
 *      successful completion of a read operation. This tag applies to
 *      H5SC_chunk_t structures.
 *
 *  H5SC_TAG_UNPIN_READ_DONE_ERROR – Indicates that a chunk was unpinned
 *      while handling an error during completion of a read operation. This
 *      tag applies to H5SC_chunk_t structures.
 *
 *  H5SC_TAG_UNPIN_ERASE_DONE – Indicates that a chunk was unpinned after
 *      successful completion of an erase operation. This tag applies to
 *      H5SC_chunk_t structures.
 *
 *  H5SC_TAG_UNPIN_ERASE_DONE_ERROR – Indicates that a chunk was unpinned
 *      while handling an error during completion of an erase operation. This
 *      tag applies to H5SC_chunk_t structures.
 *
 *  H5SC_TAG_TEST – Catch-all tag reserved for SCC testing and
 *      test-specific validation where one of the operational tags above is
 *      not appropriate.
 *
 ******************************************************************************/
typedef enum H5SC_tag_t {
    H5SC_TAG_NONE = 0,
    H5SC_TAG_CREATE,
    H5SC_TAG_HT_INSERT,
    H5SC_TAG_HT_DELETE,
    H5SC_TAG_LRU_INSERT,
    H5SC_TAG_LRU_REMOVE,
    H5SC_TAG_LRU_PROMOTE,
    H5SC_TAG_LRU_TOUCH,
    H5SC_TAG_FLUSH_DIRTY,
    H5SC_TAG_FLUSH,
    H5SC_TAG_EVICT,
    H5SC_TAG_LOOKUP,
    H5SC_TAG_PIN_IOINIT,
    H5SC_TAG_PIN_IOINIT_CACHED,
    H5SC_TAG_UNPIN_WRITE_DONE,
    H5SC_TAG_UNPIN_WRITE_DONE_ERROR,
    H5SC_TAG_UNPIN_READ_DONE,
    H5SC_TAG_UNPIN_READ_DONE_ERROR,
    H5SC_TAG_UNPIN_ERASE_DONE,
    H5SC_TAG_UNPIN_ERASE_DONE_ERROR,
    H5SC_TAG_TEST /* catch-all for testing */
} H5SC_tag_t;

/******************************************************************************
 *
 * Enumeration: H5SC_size_est_source_t
 *
 * Info
 *
 *      The enumeration identifies the source of a resident-size estimate
 *      produced for a nonresident chunk before that chunk is admitted for
 *      materialization by the SCC.
 *
 *      Resident-size estimates are used by SCC batching and active-limit
 *      admission logic to predict the final decoded allocation associated
 *      with a selected chunk. These estimates are heuristics and are not
 *      authoritative cache-accounting values. After materialization, the
 *      actual layout-reported resident allocation replaces the estimate for
 *      cache accounting and may be used to update dataset-local estimate
 *      history.
 *
 *      The source value is stored with the pending estimate in
 *      H5SC_io_sel_chunk_t so that estimate accuracy can be attributed to the
 *      mechanism that produced it.
 *
 *      Not every source represented by this enumeration is currently active.
 *      The present implementation uses dataset history when observations are
 *      available and otherwise falls back to the dense logical chunk size.
 *      Lookup-based and sparse-model estimates are reserved for future use.
 *
 * Values
 *
 *  H5SC_SIZE_EST_NONE – Indicates that no resident-size estimate source
 *      is currently associated with the selected chunk. This is the default
 *      value and is restored when a pending estimate is cleared.
 *
 *  H5SC_SIZE_EST_LOOKUP – Indicates that the estimate was derived from
 *      layout lookup information that provides or can reliably predict the
 *      complete decoded resident allocation. This source is reserved for
 *      future use by the current structured-chunk implementation.
 *
 *  H5SC_SIZE_EST_SPARSE_MODEL – Indicates that the estimate was produced
 *      by a sparse-representation model derived from information about the
 *      chunk's defined values or representation. This source is reserved for
 *      future use by the current implementation.
 *
 *  H5SC_SIZE_EST_DATASET_HISTORY – Indicates that the estimate was derived
 *      from previously observed resident allocation sizes for chunks in the
 *      same dataset. The current implementation uses the dataset-local
 *      exponential moving average and decaying high-water observation to
 *      produce this estimate.
 *
 *  H5SC_SIZE_EST_DENSE_FALLBACK – Indicates that the dataset's dense
 *      logical chunk size was used as the estimate because no usable
 *      dataset-local resident-size history was available.
 *
 ******************************************************************************/
typedef enum H5SC_size_est_source_t {
    H5SC_SIZE_EST_NONE = 0,
    H5SC_SIZE_EST_LOOKUP,
    H5SC_SIZE_EST_SPARSE_MODEL,
    H5SC_SIZE_EST_DATASET_HISTORY,
    H5SC_SIZE_EST_DENSE_FALLBACK
} H5SC_size_est_source_t;

typedef struct H5SC_stats_t        H5SC_stats_t;
typedef struct H5SC_t              H5SC_t;
typedef struct H5SC_dset_header_t  H5SC_dset_header_t;
typedef struct H5SC_chunk_t        H5SC_chunk_t;
typedef struct H5SC_chunk_key_t    H5SC_chunk_key_t;
typedef struct H5SC_io_sel_chunk_t H5SC_io_sel_chunk_t;
typedef struct H5SC_io_scratch_t   H5SC_io_scratch_t;
typedef struct H5SC_io_info_t      H5SC_io_info_t;

/*
 * Layout callbacks
 */
/* Looks up count chunk address and size on disk. defined_values_size is the number of bytes to read if
 * only the list of defined values is needed. size_hint is the suggested allocation size for the chunk
 * (could be larger if the chunk might expand when decoded). defined_values_size_hint is the suggested
 * allocation size if only the list of defined values is needed. If *defined_values_size is returned as 0,
 * then all values are defined for the chunk. In this case, the chunk may still be decoded without reading
 * from disk, by allocating a buffer of size defined_valued_size_hint and passing it to
 * H5SC_chunk_decode_t with *nbytes set to 0. *udata can be set to anything and will be passed through to
 * H5SC_chunk_decode_t and/or the selection or vector I/O routines, then freed with free() (we will create
 * an H5SC_free_udata_t callback if necessary).
 */
typedef herr_t (*H5SC_chunk_lookup_t)(struct H5D_t *dset, size_t count, const hsize_t *scaled[] /*in*/,
                                      haddr_t *addr[] /*out*/, hsize_t *size[] /*out*/,
                                      hsize_t *defined_values_size[] /*out*/, size_t *size_hint[] /*out*/,
                                      size_t *defined_values_size_hint[] /*out*/, void **udata /*out*/);

/* Decompresses/decodes the chunk from file format to memory cache format if necessary. Reallocs chunk buffer
 * if necessary. On entry, nbytes is the number of bytes used in the chunk buffer. On exit, it shall be set to
 * the total number of bytes used (not allocated) across all buffers for this chunk. On entry, alloc_size is
 * the size of the chunk buffer. On exit, it shall be set to the total number of bytes allocated across all
 * buffers for this chunk. Optional, if not present, chunk is the same in cache as on disk. partial_bound is
 * true if the chunk was encoded with partial_bound set to true. If the dataset reported
 * partial_bound_chunks_different_encoding as false, the setting of partial_bound is undefined. */
typedef herr_t (*H5SC_chunk_decode_t)(H5D_t *dset, size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                                      bool partial_bound, void **chunk /*in,out*/, void *udata);

/* The same as H5SC_chunk_decode_t but only decodes the defined values. Optional, if not present, the entire
 * chunk must always be decoded. */
typedef herr_t (*H5SC_chunk_decode_defined_values_t)(H5D_t *dset, size_t *nbytes /*in,out*/,
                                                     size_t *alloc_size /*in,out*/, bool partial_bound,
                                                     void **chunk /*in,out*/, void *udata);

/* Creates a new empty chunk. Does not insert into on disk chunk index. If fill is true, writes the fill value
 * to the chunk (unless this is a sparse chunk). The number of bytes used is returned in *nbytes and the size
 * of the chunk buffer is returned in *buf_size. */
typedef herr_t (*H5SC_new_chunk_t)(H5D_t *dset, bool fill, size_t *nbytes /*out*/, size_t *buf_size /*out*/,
                                   void **chunk /*chunk*/, void **udata /*out*/);

/* Reallocates buffers as necessary so the total allocated size of buffers for the chunk (alloc_size) is equal
 * to the total number of bytes used (nbytes). Optional, if not present the chunk cache will be more likely to
 * evict chunks if there is wasted space in the buffers. */
typedef herr_t (*H5SC_chunk_condense_t)(H5D_t *dset, size_t *nbytes /*in, out*/, void **chunk /*in, out*/,
                                        void *udata);

/* Compresses/encodes the chunk as necessary. If chunk is the same as cache_buf, leaves *write_buf as NULL.
 * This function leaves chunk alone and allocates write_buf if necessary to hold compressed data, sets
 * *write_size to the size of the data in write_buf, and sets *write_size_alloc to the size of write_buf, if
 * it was allocated. partial_bound is true if the chunk is partially outside the bounds of the dataset. If the
 * dataset reported partial_bound_chunks_different_encoding as false, the setting of partial_bound is
 * undefined. */
typedef herr_t (*H5SC_chunk_encode_t)(H5D_t *dset, hsize_t *write_size /*out*/,
                                      hsize_t *write_buf_alloc /*out*/, bool partial_bound, const void *chunk,
                                      void *udata, void **write_buf /*out*/);

/* The same as H5SC_chunk_encode_t but does not preserve chunk buffer, encoding is performed in-place. Must
 * free all other data used. */
typedef herr_t (*H5SC_chunk_encode_in_place_t)(H5D_t *dset, size_t *write_size /*out*/, bool partial_bound,
                                               void **chunk /*in,out*/, void *udata);

/* Frees chunk and all memory referenced by it. Optional, if not present free() is simply used. */
typedef herr_t (*H5SC_chunk_evict_t)(H5D_t *dset, void *chunk, void *udata);

/* Inserts (or reinserts) count chunks into the chunk index if necessary. Old address and size (if any) of the
 * chunks on disk are passed as addr and old_disk_size, the new size is passed in as new_disk_size. This
 * function resizes and reallocates on disk if necessary, returning the address of the chunks on disk in
 * *addr. */
typedef herr_t (*H5SC_chunk_insert_t)(H5D_t *dset, size_t count, const hsize_t *scaled[] /*in*/,
                                      haddr_t *addr[] /*in,out*/, hsize_t old_disk_size[],
                                      hsize_t new_disk_size[], void *chunk[] /*in*/, void *udata[]);

/* Called when the chunk cache wants to read data directly from the disk to the user buffer via selection I/O.
 * If not possible due to compression, etc, returns select_possible=false. Otherwise transforms the file space
 * if necessary to describe the selection in the on disk format (returns transformed space in file_space_out).
 * If no transformation is necessary, leaves *file_space_out as NULL. chunk may be passed as NULL, and may
 * also be an in-cache chunk that only contains information on defined values. If chunk is passed as NULL and
 * the callback requires a chunk to be passed with (at least) the defined values selection, this callback
 * shall return *require_values=true and *file_space_out=NULL. Optional, if not present, chunk I/O is only
 * performed on entire chunks or with vector I/O. The H5SC code checks for type conversion before calling
 * this. partial_bound is true if the on-disk chunk was encoded with partial_bound set to true. If the dataset
 * reported partial_bound_chunks_different_encoding as false, the setting of partial_bound is undefined. */
typedef herr_t (*H5SC_chunk_selection_read_t)(H5D_t *dset, const H5S_t *file_space_in, bool partial_bound,
                                              void *chunk /*in*/, H5S_t **file_space_out /*out*/,
                                              bool *select_possible /*out*/, bool *require_values /*out*/,
                                              void *udata);

/*
 * Called when the chunk cache wants to describe a direct read from the
 * on-disk chunk representation as a vector of file offsets and byte sizes.
 *
 * file_space_in describes the selected elements in logical chunk coordinates.
 * addr is the on-disk base address of the chunk. If the requested selection
 * cannot be represented for direct vector I/O, for example because the
 * on-disk representation requires decoding or filtering, the callback sets
 * *vector_possible to false.
 *
 * When vector I/O is possible, the callback sets *vector_possible to true,
 * returns the number of file vectors in *vec_count, and allocates/populates
 * *offsets and *sizes. Returned offsets are absolute file offsets rather than
 * offsets relative to the beginning of the chunk.
 *
 * chunk may be NULL or may reference a decoded object containing at least
 * defined-value information. If the callback cannot determine the requested
 * vectors without defined-value state that is not currently available, it
 * sets *require_values to true and does not return a usable vector.
 *
 * This callback is optional. If it is not provided, the SCC must use another
 * supported I/O path.
 *
 * The SCC must verify that direct vector I/O is compatible with any required
 * datatype conversion before using the returned vectors.
 *
 * partial_bound is true when the on-disk chunk was encoded as a partial-edge
 * chunk. If the layout reports partial_bound_chunks_different_encoding as
 * false, the value of partial_bound is not significant.
 */
typedef herr_t (*H5SC_chunk_vector_read_t)(H5D_t *dset, haddr_t addr, const H5S_t *file_space_in,
                                           bool partial_bound, void *chunk /*in*/, size_t *vec_count /*out*/,
                                           haddr_t **offsets /*out*/, size_t **sizes /*out*/,
                                           bool *vector_possible /*out*/, bool *require_values /*out*/,
                                           void *udata);

/* Called when the chunk cache wants to write data directly from the user buffer to the cache via selection
 * I/O. If not possible due to compression, etc, returns select_possible=false. Otherwise transforms the file
 * space if necessary to describe the selection in the on disk format (returns transformed space in
 * file_space_out). If no transformation is necessary, leaves *file_space_out as NULL. chunk may be passed as
 * NULL, and may also be an in-cache chunk that only contains information on defined values. If chunk is
 * passed as NULL and the callback requires a chunk to be passed with (at least) the defined values selection,
 * this callback shall return *require_values=true and *file_space_out=NULL. Optional, if not present, chunk
 * I/O is only performed on entire chunks or with vector I/O. The H5SC code checks for type conversion before
 * calling this. partial_bound is true if the on-disk chunk was encoded with partial_bound set to true. If the
 * dataset reported partial_bound_chunks_different_encoding as false, the setting of partial_bound is
 * undefined.
 * NOTE: When implementing this function, consider whether selection_write should be an H5S_t ** in order to
 *       match the semantics of H5SC_chunk_selection_read_t.
 */
typedef herr_t (*H5SC_chunk_selection_write_t)(H5D_t *dset, const H5S_t *file_space_in, bool partial_bound,
                                               void *chunk /*in*/, H5S_t *file_space_out /*out*/,
                                               bool *select_possible /*out*/, bool *require_values /*out*/,
                                               void *udata);

/*
 * Called when the chunk cache wants to describe a direct write to the
 * on-disk chunk representation as a vector of file offsets and byte sizes.
 *
 * file_space_in describes the selected elements in logical chunk coordinates.
 * addr is the on-disk base address of the chunk. If the requested selection
 * cannot be represented for direct vector I/O, for example because the
 * on-disk representation requires encoding or filtering, the callback sets
 * *vector_possible to false.
 *
 * When vector I/O is possible, the callback sets *vector_possible to true,
 * returns the number of file vectors in *vec_count, and allocates/populates
 * *offsets and *sizes. Returned offsets are absolute file offsets rather than
 * offsets relative to the beginning of the chunk.
 *
 * chunk may be NULL or may reference a decoded object containing at least
 * defined-value information. If the callback cannot determine the requested
 * vectors without defined-value state that is not currently available, it
 * sets *require_values to true and does not return a usable vector.
 *
 * This callback is optional. If it is not provided, the SCC must use another
 * supported I/O path.
 *
 * The SCC must verify that direct vector I/O is compatible with any required
 * datatype conversion before using the returned vectors.
 *
 * partial_bound is true when the on-disk chunk was encoded as a partial-edge
 * chunk. If the layout reports partial_bound_chunks_different_encoding as
 * false, the value of partial_bound is not significant.
 */
typedef herr_t (*H5SC_chunk_vector_write_t)(H5D_t *dset, haddr_t addr, const H5S_t *file_space_in,
                                            bool partial_bound, void *chunk /*in*/, size_t *vec_count /*out*/,
                                            haddr_t **offsets /*out*/, size_t **sizes /*out*/,
                                            bool *vector_possible /*out*/, bool *require_values /*out*/,
                                            void *udata);

/* Scatters data from the chunk buffer into the memory buffer (in dset_info), performing type conversion if
 * necessary. file_space's extent matches the chunk dimensions and the selection is within the chunk.
 * mem_space's extent matches the entire memory buffer's and the selection within it is the selected values
 * within the chunk, offset appropriately within the full extent. Optional, if not present, chunk is the same
 * in memory as it is in cache, with the exception of type conversion (which will be handled by the H5SC
 * layer). If the layout stores variable length data within the chunk this callback must be defined. */
typedef herr_t (*H5SC_chunk_scatter_mem_t)(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t *io_type_info,
                                           const H5S_t *mem_space, const H5S_t *file_space, const void *chunk,
                                           void *udata);

/* Gathers data from the memory buffer (in dset_info) into the chunk buffer, performing type conversion if
 * necessary. file_space's extent matches the chunk dimensions and the selection is within the chunk.
 * mem_space's extent matches the entire memory buffer's and the selection within it is the selected values
 * within the chunk, offset appropriately within the full extent. Defines selected values in the chunk.
 * alloc_size_total represents the total number of bytes allocated across all buffers for this chunk, and must
 * be updated by this callback if that size changes. Optional, if not present, chunk is the same in memory as
 * it is in cache, with the exception of type conversion (which will be handled by H5SC layer). If the layout
 * stores variable length data within the chunk this callback must be defined. */
typedef herr_t (*H5SC_chunk_gather_mem_t)(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t *io_type_info,
                                          const H5S_t *mem_space, const H5S_t *file_space,
                                          size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                                          size_t *alloc_size_total /*in,out*/, void *chunk, void *udata);

/* Propagates the fill value into the selected elements of the chunk buffer, performing type conversion if
 * necessary. space's extent matches the chunk dimensions and the selection is within the chunk.
 * alloc_size_total represents the total number of bytes allocated across all buffers for this chunk, and must
 * be updated by this callback if that size changes. Optional, if not present, chunk is the same in memory as
 * it is in cache, with the exception of type conversion (which will be handled by H5SC layer). If the layout
 * stores variable length data within the chunk this callback must be defined. */
typedef herr_t (*H5SC_chunk_fill_t)(H5D_dset_io_info_t *dset_info, H5D_io_type_info_t *io_type_info,
                                    H5S_t *space, size_t *nbytes /*in,out*/, size_t *alloc_size /*in,out*/,
                                    size_t *alloc_size_total /*in,out*/, void *chunk, void *udata);

/* Queries the defined elements in the chunk. selection may be passed as H5S_ALL. These selections are within
 * the logical chunk. Optional, if not present, all values are defined. */
typedef herr_t (*H5SC_chunk_defined_values_t)(H5D_t *dset, const H5S_t *selection, void *chunk,
                                              H5S_t **defined_values /*out*/, void *udata);

/* Erases the selected elements in the chunk, causing them to no longer be defined. If all values in the chunk
 * are erased and the chunk should be deleted, sets *delete_chunk to true, causing the cache to delete the
 * chunk from cache, free it in memory using H5SC_chunk_evict_t, and delete it on disk using
 * H5SC_chunk_delete_t. These selections are within the logical chunk. Optional, if not present, the fill
 * value will be written to the selection using H5SC_chunk_fill_t. */
typedef herr_t (*H5SC_chunk_erase_values_t)(H5D_t *dset, const H5S_t *selection, size_t *nbytes /*in,out*/,
                                            size_t *alloc_size /*in,out*/, void *chunk,
                                            bool *delete_chunk /*out*/, void *udata);

/* Frees the data values in the cached chunk and memory used by them (but does not reallocate - see
 * H5SC_chunk_condense_t), but leaves the defined values intact. Optional, if not present the entire chunk
 * will be evicted. */
typedef herr_t (*H5SC_chunk_evict_values_t)(H5D_t *dset, size_t *nbytes /*in,out*/,
                                            size_t *alloc_size /*in,out*/, void *chunk, void *udata);

/* Queries data about the dataset from the layout client. The callback shall set the chunk dimensions in the
 * chunk_dims array (the number of dimensions is the same as the rank of the dataset), whether encoding and
 * decoding is necessary for chunks between cache and disk, and shall set whether chunks that are partially
 * outside the bounds of the dataset are encoded differently (for example, they may not have filters applied).
 * If *partial_bound_chunks_different_encoding is set to true, then chunks whose partial bound state changes
 * will be re-encoded and re-inserted as necessary after the dataset extent changes to ensure they are encoded
 * appropriately. */
typedef herr_t (*H5SC_layout_query)(H5D_t *dset, hsize_t *chunk_dims, bool *encode_decode_necessary,
                                    bool *partial_bound_chunks_different_encoding);

/* Removes the chunk from the index and deletes it on disk. Only called if a chunk goes out of scope due to
 * H5Dset_extent() or if H5SC_chunk_erase_values_t returns *delete_chunk == true. */
typedef herr_t (*H5SC_delete_chunk_t)(H5D_t *dset, const hsize_t *scaled /*in*/, haddr_t addr,
                                      hsize_t disk_size);

/* Operations that are implemented by shared chunk cache clients */
struct H5SC_layout_ops_t {
    H5SC_chunk_lookup_t                lookup;
    H5SC_chunk_decode_t                decode;
    H5SC_chunk_decode_defined_values_t decode_defined_values;
    H5SC_new_chunk_t                   new_chunk;
    H5SC_chunk_condense_t              condense;
    H5SC_chunk_encode_t                encode;
    H5SC_chunk_encode_in_place_t       encode_in_place;
    H5SC_chunk_evict_t                 evict;
    H5SC_chunk_insert_t                insert;
    H5SC_chunk_selection_read_t        selection_read;
    H5SC_chunk_vector_read_t           vector_read;
    H5SC_chunk_selection_write_t       selection_write;
    H5SC_chunk_vector_write_t          vector_write;
    H5SC_chunk_scatter_mem_t           scatter_mem;
    H5SC_chunk_gather_mem_t            gather_mem;
    H5SC_chunk_fill_t                  fill;
    H5SC_chunk_defined_values_t        defined_values;
    H5SC_chunk_erase_values_t          erase_values;
    H5SC_chunk_evict_values_t          evict_values;
    H5SC_layout_query                  layout_query;
    H5SC_delete_chunk_t                delete_chunk;
};

/*****************************/
/* Library-private Variables */
/*****************************/

/***************************************/
/* Library-private Function Prototypes */
/***************************************/

/* Functions that operate on a shared chunk cache */
H5_DLL H5SC_t *H5SC_create(H5F_t *file, H5P_genplist_t *fa_plist, H5SC__cache_config_t *config_ptr);
H5_DLL herr_t  H5SC_destroy(H5SC_t *cache);

/* Flush functions */
H5_DLL herr_t H5SC_flush(H5SC_t *cache);
H5_DLL herr_t H5SC_flush_dset(H5SC_t *cache, H5D_t *dset, bool evict_after_flush);

/* I/O functions */
H5_DLL herr_t H5SC_invoke_write(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info);
H5_DLL herr_t H5SC_invoke_read(H5SC_t *cache, size_t count, H5D_dset_io_info_t *dset_info);
H5_DLL herr_t H5SC_read(H5SC_t *cache, H5D_dset_io_info_t *dset_info);
H5_DLL herr_t H5SC_write(H5SC_t *cache, H5D_dset_io_info_t *dset_info);
H5_DLL herr_t H5SC_direct_chunk_read(H5SC_t *cache, H5D_t *dset, const hsize_t *offset, void *udata,
                                     void *buf, size_t *buf_size);
H5_DLL herr_t H5SC_direct_chunk_write(H5SC_t *cache, H5D_t *dset, const hsize_t *offset, void *udata,
                                      const void *buf);
H5_DLL H5S_t *H5SC_get_defined(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space);
H5_DLL herr_t H5SC_erase(H5SC_t *cache, H5D_t *dset, const H5S_t *file_space);

/* Other functions */
H5_DLL herr_t H5SC_set_extent_notify(H5SC_t *cache, H5D_t *dset, const hsize_t *old_dims);
H5_DLL herr_t H5SC_validate_config(const H5SC__cache_config_t *config_ptr);

/* Dataset specific helper functions */

H5_DLL H5SC_dset_header_t *H5SC_dset_create_header(H5SC_t *cache, haddr_t addr, size_t max_chunk_size);
H5_DLL herr_t              H5SC_dset_destroy_header(H5SC_t *cache, H5D_t *dset, H5SC_dset_header_t *dset_hdr);

H5_DLL herr_t  H5SC_dset_is_empty(H5SC_dset_header_t *dset_hdr, bool *is_empty);
H5_DLL haddr_t H5SC_dset_get_addr(H5SC_dset_header_t *dset_hdr);

#if H5SC_DO_SANITY_CHECKS
H5_DLL herr_t H5SC__test_chunk_pin(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk);
H5_DLL herr_t H5SC__test_chunk_unpin(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk);
H5_DLL herr_t H5SC__test_chunk_set_dirty(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, H5SC_chunk_t *chunk,
                                         bool dirty);
H5_DLL herr_t H5SC__test_ensure_space(H5SC_t *cache, size_t bytes_needed);
H5_DLL herr_t H5SC__test_ensure_oversized_single_chunk_space(H5SC_t *cache, size_t bytes_needed);
H5_DLL herr_t H5SC__test_trim_to_quiescent_limit(H5SC_t *cache);
H5_DLL herr_t H5SC__test_account_chunk_link_change(H5SC_t *cache, H5SC_dset_header_t *dset_hdr,
                                                   size_t old_dset_size);
#endif

/* Hash Table Functions
 * Heads are stored on H5SC_t; these helpers only link/unlink.
 * They never free or mutate payload fields.
 */

/* Hash table specific functions */
void H5SC__hash_init(H5SC_t *cache);
void H5SC__reset_hash_tables(H5SC_t *cache); /* unlink-all (no free), set heads NULL */

H5SC_chunk_t *H5SC__ht_chunk_find(H5SC_t *cache, const H5SC_chunk_key_t *chk_key);
herr_t        H5SC__ht_chunk_insert(H5SC_t *cache, H5SC_chunk_t *chk);
herr_t        H5SC__ht_chunk_delete(H5SC_t *cache, const H5SC_chunk_key_t *chk_key);

H5SC_dset_header_t *H5SC__ht_dset_find(H5SC_t *cache, haddr_t addr);
herr_t              H5SC__ht_dset_insert(H5SC_t *cache, H5SC_dset_header_t *dset_hdr);
herr_t              H5SC__ht_dset_delete(H5SC_t *cache, haddr_t addr);

herr_t H5SC__drop_dset_chunks_for_test(H5SC_t *cache, H5D_t *dset);

/* Dataset specific DLL functions */
herr_t H5SC__dset_lru_prepend(H5SC_t *cache, struct H5SC_dset_header_t *dset_hdr);
herr_t H5SC__dset_lru_promote(H5SC_t *cache, H5SC_dset_header_t *dset_hdr);
herr_t H5SC__dset_lru_remove(H5SC_t *cache, struct H5SC_dset_header_t *dset_hdr);

/* Chunk specific DLL functions */
herr_t H5SC__chunk_lru_prepend(H5SC_t *cache, struct H5SC_dset_header_t *dset_hdr,
                               struct H5SC_chunk_t *chunk);
herr_t H5SC__chunk_lru_remove(H5SC_t *cache, struct H5SC_dset_header_t *dset_hdr, struct H5SC_chunk_t *chunk);

/* Internal size operations */
herr_t H5SC__chunk_update_cached_size(H5SC_t *cache, H5SC_dset_header_t *dset_hdr, struct H5SC_chunk_t *chunk,
                                      size_t new_size);

H5SC_chunk_t *H5SC__make_chunk(H5SC_chunk_key_t key, size_t cached_sz, size_t counter, bool pio);

herr_t H5SC__get_cache_from_file_id(hid_t file_id, H5SC_t **cache);

#endif /* H5SCprivate_H */
