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

/*
 * Purpose:     This file contains declarations which are visible only within
 *              the H5SC package.  Source files outside the H5SC package should
 *              include H5SCprivate.h instead.
 */
#if !(defined H5SC_FRIEND || defined H5SC_MODULE)
#error "Do not include this file outside the H5SC package!"
#endif

#ifndef H5SCpkg_H
#define H5SCpkg_H

/* To enable dumps, add this build flag: -DH5SC_ENABLE_DUMPS=1*/

/* Get package's private header */
#include "H5SCprivate.h"
#include "H5VMprivate.h"
#include "H5Dprivate.h"
#include "H5Iprivate.h"
#include "H5private.h" /* Included for uthash access through the H5 wrapper*/

#ifdef H5SC_DO_SANITY_CHECKS
#error "Do not define H5SC_DO_SANITY_CHECKS via compiler flags. Edit directly H5SCpkg.h instead."
#endif

/* Change this value to 1 for sanity checks */
#define H5SC_DO_SANITY_CHECKS 0
/* Other private headers needed by this file */

/* Validate the sanity check value early. */
#if !((H5SC_DO_SANITY_CHECKS == 0) || (H5SC_DO_SANITY_CHECKS == 1))
#error "The value of H5SC_DO_SANITY CHECKS must be 0 or 1."
#endif

/* Magic values for structures */
#define H5SC_DSET_HDR_MAGIC UINT32_C(0x53434448) /* 'SCDH' */
#define H5SC_CHUNK_MAGIC    UINT32_C(0x53434348) /* 'SCCH' */

/**************************/
/* Package Private Macros */
/**************************/

/* Macros for the SCC; Hash tables, DLLs, etc.? */

/****************************/
/* Package Private Typedefs */
/****************************/

/******************************************************************************
 *
 * Structure: H5SC_chunk_key_t
 *
 * Info
 *
 * 	The structured stores a unique 128-bit key for a chunk. The key is derived
 *	from the dataset’s object header address in the file and the chunk’s
 *	serialized logical coordinates using an LSB-first bit interleaving schema.
 *
 * Fields
 *
 * 	uint64_t low_half  – The bottom 64-bits of the 128-bit integer key.
 *
 * 	uint64_t high_half – The top 64-bits of the 128-bit integer key.
 *
 * 	Example: Assume 64-bit dataset object header address = 610 or …01102,
 *             and 64-bit chunk logical coordinate = 510 or …01012.
 *             LSB-first bit interleaving yields to 0,1,1,0,1,1 → 01101100…2
 *		   with remaining zeros omitted for clarity. The result of
 *             the operation is stored in the two 64-bit integer fields
 *             high_half = 01101100…2; low_half = 00000000…2;
 *
 ******************************************************************************/

struct H5SC_chunk_key_t {
    uint64_t high_half;
    uint64_t low_half;
};

/******************************************************************************
 *
 * Structure: H5SC_chunk_t
 *
 * Info
 *
 * 	The structure represents an individual chunk entry in SCC; it is an entry
 *    in the chunk hash table.
 *
 *	The structure stores the chunk key, pointers to in-memory and on disk chunk
 *	data, chunk size, state flags, and pointers necessary to track the chunk as
 *	a member of the associated dataset’s chunk LRU list. A hash handle is
 *	included to make the structure hashable.
 *
 * Fields
 *
 * 	uint64_t magic – An integer value that is set by internal functions that
 *		create an instance of H5SC_chunk_t structure. Expected to be set to be
 *		1396917064. This value is used to verify that an instance of a
 *		H5SC_chunk_t structure was created properly.
 *
 *	H5SC_chunk_key_t data_key – The unique key associated with the cached
 *		chunk. It is primarily used for chunk lookups through the chunk-
 *		focused hash table and operations that require internal chunk index
 *		updates.
 *
 *	void *chunk_obj – Pointer to the decoded chunk object, which is an
 *		H5D_chunk_cache_mem_t structure. Necessary for callback operations.
 *
 * 	void *udata – Pointer to the chunk udata, which is an H5D_chunk_ud_t
 *		structure. Necessary for callback operations.
 *
 *
 *	unsigned ndims – Number of valid dims defined in the scaled coordinates for
 *		this chunk. The default value is 0, a convenient invalid value.
 *
 *	hsize_t chunk_log_coord – The linearized, dataset-relative chunk coordinate,
 *		referred to as the chunk logical coordinate (index). This value is
 *		computed from the dataset dimensions, chunk dimensions, and element
 *		position. It is computed when a H5SC_chunk_t structure is added to the
 *		SCC and is used as one of the components for the chunk key
 *		(H5SC_chunk_key_t). This value is stored as a reference for checking
 *		whether a chunk should be reindexed when a dataset contains one or
 *		more extensible dimensions.
 *
 *	hsize_t scaled[H5S_MAX_RANK] – The scaled coordinates associated with this
 *		chunk (essentially the offset divided by the chunk dimension (for each
 *		rank)). This array is set during the initialization of an I/O request
 *		and is used for cache flush procedures and callback operations.
 *
 *	haddr_t disk_addr – The on-disk address of the chunk. Set to HADDR_UNDEF
 *		when the chunk is created. Necessary for cache hits to be processed
 *		correctly.
 *
 *	size_t cached_chunk_size – The size of the in-memory data buffer after
 *		decoding the chunk and the application of any required filters. This
 *		field is updated when the in-memory data is altered. The default
 *		value is set to 0.
 *
 *	size_t disk_nbytes – The last known size of the on disk data buffer prior to
 *		applying filters or decoding. It is obtained using the
 *		H5D__struct_chunk_lookup callback. The default value is set to 0.
 *
 *	size_t chunk_counter – A priority counter primarily used with eviction
 *		policies. It is set to 0 by default. In the future, it may be used to
 *		identify eviction candidates based on access history or other encoded
 *		states. For example, during eviction candidate selection, if the
 *		partial_io flag is set to True, this counter would be incremented to a
 *		value of 1 to indicate that it has been flagged for a second pass
 *		through the LRU list.
 *
 * 	H5SC_tag_t last_tag – Used to indicate what the most recent internal
 *		operation was that interacted with this chunk. Set to H5SC_TAG_CREATE
 *		when a H5SC_chunk_t structure is created. See the section defining
 *		H5SC_tag_t for possible values.
 *
 *	bool dirty_flag – A flag to indicate when an in-memory chunk buffer has been
 *		modified. A value of True indicates a chunk buffer is dirty. This flag
 *		is set when a chunk is created/modified/resized. Set to False by
 *		default.
 *
 *	bool partial_io – A flag to indicate if a partial read or write operation
 *		has been done on this chunk (i.e., the intersection of the selection
 *		describing this chunk and the selection on the dataset resulted in
 *		less than the whole chunk being selected). It is used to track I/O
 *		states and inform eviction policies. This field is set to false by
 *		default.
 *
 *	struct H5SC_chunk_t *prev_ptr – The pointer to the previous node in the
 *		dataset’s LRU list. It is Null if this chunk is at the head of the LRU
 *		list or if the LRU list has a single element.
 *
 *	struct H5SC_chunk_t *next_ptr – The pointer to the next node in the
 *		dataset’s LRU list. It is Null if this chunk is at the tail of the LRU
 *		list or if the LRU list has a single element.
 *
 *	UT_hash_handle hval_chunk – A hash handle is required for this structure to
 *		be hashable by UTHash; it is used to track this chunk in the SCC chunk
 *		hash table. Set to Null by default.
 *
 ******************************************************************************/

struct H5SC_chunk_t {
    uint64_t             magic;
    H5SC_chunk_key_t     data_key;
    void                *chunk_obj; /* Decoded chunk object (H5D_chunk_cache_mem_t *) */
    void                *udata;     /* Chunk udata (H5D_chunk_ud_t *) */
    unsigned             ndims;     /* Number of valid dims in scaled[] */
    hsize_t              chk_log_coord;
    hsize_t              scaled[H5S_MAX_RANK]; /* Persistent scaled coordinates for lookup/flush logic */
    haddr_t              disk_addr;            /* On-disk address of the chunk */
    size_t               cached_chunk_size;
    size_t               disk_nbytes; /* Formally disk_chunk_size; last know encoded size */
    size_t               chunk_counter;
    H5SC_tag_t           last_op;
    bool                 dirty_flag;
    bool                 partial_IO;
    struct H5SC_chunk_t *prev_ptr;
    struct H5SC_chunk_t *next_ptr;
    UT_hash_handle       hval_chunk;
};

/******************************************************************************
 *
 * Structure: H5SC_io_info_t
 *
 * Info
 *
 * 	The structure represents the state associated with a dataset I/O request
 *    in SCC. It encapsulates the set of chunks participating in the operation
 *    and provides bookkeeping fields to manage iteration and storage of those
 *    selected chunks.
 *
 *	The structure primarily serves as a container for chunk selection results
 *    produced during I/O setup. It is used to track which chunks are involved in
 *	an I/O operation, the starting position (used for batching), how many chunks
 *	are present in the request, and how large sel_chunks currently is (in
 *	bytes). These fields are subsequently consume by I/O execution paths and
 *	chunk processing routines.
 *
 * Fields
 *
 * 	H5SC_io_sel_chunk_t *sel_chunks – Pointer to a vector of structures
 *		describing the chunks selected for the current I/O operation. Each
 *		element contains per-chunk selection and metadata required to
 *		execute I/O on that chunk. Default size of 128 and is dependent on
 *		the size of sel_chunk_alloc.
 *
 *	size_t sel_start – The starting index within sel_chunks for processing.
 *		This field is typically used to support batched processing of selected
 *		chunks.
 *
 *	size_t num_sel_chunks – The number of valid chunk entries currently
 *		stored in sel_chunks. This represents the number of chunks
 *		participating in the I/O request.
 *
 *	size_t sel_chunks_alloced – The total number of elements allocated for
 *		sel_chunks. This value defines the current capacity of the array
 *		and is used to determine when reallocation is required as more
 *		chunks are selected. This value is 128 by default and increases by
 *		multiples of 2 as necessary.
 *
 ******************************************************************************/
struct H5SC_io_info_t {
    H5SC_io_sel_chunk_t *sel_chunks;
    size_t               sel_start;
    size_t               num_sel_chunks;
    size_t               sel_chunks_alloced;
};

/******************************************************************************
 *
 * Structure: H5SC_io_sel_chunk_t
 *
 * Info
 *
 * 	The structure represents a single chunk participating in a dataset I/O
 *    request in SCC. It encapsulates all per-chunk state required to map
 *    between dataset coordinates, chunk indexing, memory/file dataspaces,
 *    and any associated cached chunk entry used during I/O execution.
 *
 *	The structure serves as the fundamental unit of work for chunked I/O
 *    operations. It binds together spatial information (coordinates and
 *    scaled indices), cache state, and dataspace selections needed to
 *    correctly perform partial or full reads/writes on an individual chunk.
 *    Lookup-related fields are included to support lookup-based behavior.
 *
 * Fields
 *
 * 	H5D_dset_io_info_t *dset_info – Pointer to the dataset I/O context for
 *		the dataset this chunk belongs to. Provides access to layout,
 *		filter pipeline, and other dataset-level information required
 *		for I/O execution.
 *
 *	hsize_t coords[H5S_MAX_RANK] – The absolute coordinates of the chunk
 *		within the dataset’s logical space. These correspond to the
 *		chunk’s offset in each dimension.
 *
 *	hsize_t scaled[H5S_MAX_RANK] – The scaled chunk coordinates, computed
 *		by dividing coords by the chunk dimensions for each rank. These
 *		values are used for chunk indexing, hashing, and cache lookup
 *		operations.
 *
 *	hsize_t chk_log_coord – The computed logical coordinate of the chunk,
 *		uniquely identifying the chunk within the dataset’s chunk index
 *		space. This is typically derived from the scaled coordinates.
 *
 *	H5SC_chunk_t *cached_chunk – Pointer to the corresponding SCC cache
 *		entry for this chunk. This is populated when the chunk is found
 *		or inserted into the cache and is NULL if no cache entry has
 *		been associated yet.
 *
 *	bool lookup_valid – Indicates whether the lookup-related fields
 *		contain valid information for this chunk.
 *
 *	haddr_t lookup_addr – The on-disk address obtained from lookup
 *		operations.
 *
 *	hsize_t lookup_disk_nbytes – The size of the chunk on disk as
 *		returned by lookup operations.
 *
 *	hsize_t lookup_defined_values_size – Size of defined values metadata
 *		associated with the chunk from lookup operations.
 *
 *	size_t lookup_size_hint – Size hint used to optimize lookup-related
 *		operations.
 *
 *	size_t lookup_defined_values_size_hint – Size hint for defined
 *		values metadata used during lookup.
 *
 *	void *lookup_udata – Pointer to user data structure used during
 *		lookup callbacks.
 *
 *	H5S_t *file_space – Dataspace describing the selection within the
 *		chunk in file coordinates. The extent matches the chunk
 *		dimensions, and the selection represents the portion of the
 *		chunk involved in the I/O operation.
 *
 *	H5S_t *mem_space – Dataspace describing the selection within the
 *		user’s memory buffer corresponding to this chunk. The extent
 *		matches the overall memory dataspace, with the selection
 *		restricted to the elements mapped to this chunk.
 *
 *	bool file_space_shared – Flag indicating whether file_space is shared with
 *		another owner. This boolean is false by default and indicates that the
 *		selection was specific for this chunk. Set to true when the filespace
 *		is reused across multiple chunk-selection records.
 *
 *	bool mem_space_shared – Flag indicating whether mem_space is shared with
 *		another owner. This boolean is false by default and indicates that the
 *		selection was specific for this chunk. Set to true when the memspace
 *		is reused across multiple chunk-selection records.
 *
 *	H5_flexible_const_ptr_t buf – Pointer to the user-provided memory
 *		buffer used for I/O. This buffer is used as the source or
 *		destination for data transferred between memory and the chunk.
 *
 ******************************************************************************/
struct H5SC_io_sel_chunk_t {
    H5D_dset_io_info_t     *dset_info;
    hsize_t                 coords[H5S_MAX_RANK];
    hsize_t                 scaled[H5S_MAX_RANK];
    hsize_t                 chk_log_coord;
    H5SC_chunk_t           *cached_chunk;
    bool                    lookup_valid;
    haddr_t                 lookup_addr;
    hsize_t                 lookup_disk_nbytes;
    hsize_t                 lookup_defined_values_size;
    size_t                  lookup_size_hint;
    size_t                  lookup_defined_values_size_hint;
    void                   *lookup_udata;
    H5S_t                  *file_space;
    H5S_t                  *mem_space;
    bool                    file_space_shared;
    bool                    mem_space_shared;
    H5_flexible_const_ptr_t buf;
};

/******************************************************************************
 *
 * Structure: H5SC_dset_header_t
 *
 * Info
 *
 *	The header structure for a dataset with data held in the shared chunk cache.
 *	It tracks the dataset identity (represented by the dataset object header
 *	address), the minimum amount of chunk data that should be maintained in this
 *	dataset (in bytes), accounts for the total size of cached chunks. The
 *	structure contains pointers to the head and tail of the LRU list for the
 *	cached chunks in this dataset, pointers to the next and previous datasets in
 *	the dataset LRU list, and pointers which are necessary for eviction
 *	management.
 *
 * Fields
 *
 *	uint64_t magic – Value set when a H5SC_dset_header_t structure is created.
 *		This value is used for debugging to ensure that the structure is
 *		created/destroyed consistenly.
  *
 * 	haddr_t dset_addr –  The 64-bit address of the object header for this
 *		dataset. This field is used as the key for the dataset hash table.
 *		It is set to 0 by default.
 *
 *	size_t min_dset_size – Represents the minimum number of bytes to
 *		remain cached for this dataset under typical memory constraints. The
 *		default size is 10 MB and is user-configurable.
 *
 * 	size_t chunk_lru_len – The number of chunks present within the SCC. This
 * 		value is used as the running count needed for DLL operations. Set to 0
 *		by default.
 *
 *	size_t curr_dset_size – The current amount of chunk data, in bytes *
 *		associated with this dataset within the cache (i.e., is a sum of
 *		current chunk sizes). This quantity is updated when a chunk is added,
 *		removed, or resized. This field contributes to total cache
 *		utilization tracked in H5SC_t. It is set to 0 by default.
 *
 *	H5SC_tag_t last_tag – An enum-typed tag used during debugging. Tags are
 *		updated near the end of a routine. H5SC_tag_t values do not
 *		differentiate between chunks and datasets.
 *
 *	H5SC_io_info_t *io_info – Structure used for an I/O request that includes
 *		chunks from this dataset. This structure is populated by
 *		H5SC_io_info_init() and is used by H5SC_read() and H5SC_write().
 *
 *	bool resize_in_progress – A flag used to indicate that a resize operation is
 *		being conducted on this dataset. A value of True is set if dataset is
 *		being resized. While true, any internal SCC
 *		procedures and any I/O requests for this dataset should be interrupted
 *		until the resize operation is completed. It is set to false by
 *		default.
 *
 *	bool evict_exhausted – A flag used to indicate that this dataset has
 *		nominated as many eviction candidates as it is able to and should not
 *		be overlooked for future eviction candidate selections during this I/O
 *		cycle. During eviction candidate selection, this field is set to True
 *		if H5SC_chunk_t elements have been removed and the min_dset_size has
 *		been reached. When this field is True, this dataset header is removed
 *		from the dataset LRU list and is added to an exhausted dataset list
 *		through utilizing the associated pointers (*next_exhausted_ptr and
 *		*prev_exhausted_ptr). This field is set to False by default.
 *
 *	H5D_t *dset – Pointer to the dataset; included for debugging convenience.
 *
 *	H5SC_chunk_t *lru_head_ptr – The pointer to the head of this dataset’s chunk
 *		LRU list (indicating it is the most recently used chunk). Set to
 *		Null by default, when the chunk LRU list is empty, or when the chunk
 *		LRU list contains a single element.
 *
 *	H5SC_chunk_t *lru_tail_ptr – The pointer to the tail of this dataset’s chunk
 *		LRU list (indicating it is the least recently used chunk). Set to Null
 *		by default, when the chunk LRU list is empty, or when the chunk LRU
 *		list contains a single element.
 *
 *	H5SC_dset_header_t *next_dset_ptr – The pointer to the next dataset in the
 *		dataset LRU list. Set to Null by default, when this header is at the
 *		tail of the dataset LRU list, or when the list has a single element.
 *
 *	H5SC_dset_header_t *prev_dset_ptr – The pointer to the previous dataset in
 *		the dataset LRU list. Set to Null by default, when this header is at
 *		the head of the dataset LRU list, or when the list has a single
 *		element.
 *
 *	H5SC_dset_header_t *next_exhausted_ptr; - The pointer to the next dataset in
 *		the exhausted dataset linked list. Set to Null by default, when this
 *		dataset is at the exhausted list tail, or if the exhausted list has a
 *		single element.
 *
 *	H5SC_dset_header_t *prev_exhausted_ptr; - The pointer to the previous
 *		dataset in the exhausted dataset linked list. Set to Null by default,
 *		when this dataset is at the exhausted list head, or if the exhausted
 *		list has a single element.
 *
 *	UT_hash_handle dset_hval – A hash handle is required for this structure to
 *		be hashable by UTHash; it is used to track this dataset in the SCC
 *		dataset hash table. Set to Null by default.
 *
 ******************************************************************************/

struct H5SC_dset_header_t {
    uint64_t                   magic;
    haddr_t                    dset_addr;
    size_t                     min_dset_size;
    size_t                     chunk_lru_len;
    size_t                     curr_dset_size;
    H5SC_tag_t                 last_op;
    H5SC_io_info_t            *io_info;
    bool                       resize_in_progress;
    bool                       evict_exhausted;
    struct H5D_t              *dset;
    struct H5SC_chunk_t       *lru_head_ptr;
    struct H5SC_chunk_t       *lru_tail_ptr;
    struct H5SC_dset_header_t *next_dset_ptr;
    struct H5SC_dset_header_t *prev_dset_ptr;
    struct H5SC_dset_header_t *next_exhausted_ptr;
    struct H5SC_dset_header_t *prev_exhausted_ptr;
    UT_hash_handle             dset_hval;
};

/******************************************************************************
 *
 * Structure: H5SC_t
 *
 * Info
 *
 * 	The primary structure used for shared chunk cache management. Provides
 *		fields for maintaining access to the head and tail of the dataset LRU
 *		list, tracking the active and quiescent cache memory usage, and the
 *		pointers necessary to access the hash tables used for chunk and
 *		dataset lookups.
 *
 * Fields
 *
 *	size_t SCC_quiescent_size – The amount of quiescent data, in bytes, present
 *		within the cache when there are no I/O operation being performed on
 *		cached data. This field is computed using the sum of the
 *		curr_dset_size field from each dataset currently in the SCC. This
 *		field is updated whenever a chunk is added, removed, or resized while
 *		present within the SCC. A value of 0 indicates the cache is empty. A
 *		value of 0 is set by default when this structure is created.
 *
 *	size_t SCC_quiescent_limit – The amount of quiescent data, in bytes, that
 *		should be retained within the cache when there is no I/O operation
 *		being performed on cached data. This field is defined to be 1 GB by
 *		default and may be set using a FAPL configuration (using the
 *		max_q_size field in a H5SC__cache_config_t structure.)
 *
 *	size_t SCC_active_size – The amount of active data, in bytes, allocated to
 *		the SCC when an I/O request is being processed. This field is an
 *		integer-multiple of the current SCC_quiescent_size. The multiplier is
 *		user configurable and is set to be 2x the size of the quiescent limit
 *		by default to accommodate the application of filters or other
 *		operations that often cause chunks to exceed estimated sizes. Once an
 *		I/O request is completed, this field is set to 0.
 *		--Currently not properly utilized; SCC_quiescent_size became the
 *		catch-all for in-process size tracking. This
 *		in a small refactor and is reserved as an optimization at this time.
 *
 *	size_t SCC_active_limit – The amount of active data, in bytes, that can be
 *		processed at one time by the SCC. This quantity is treated as the
 *		upper limit for available memory during I/O requests. It is set to be
 *		2 GB default nd may be set using a FAPL configuration (using the
 *		max_a_size field in a H5SC__cache_config_t structure.)
 *
 *  	size_t SCC_max_bytes - The hard cap, in bytes, for resident cache size.
 * 		When SCC_quiescent_size + new_allocation would exceed SCC_max_bytes,
 * 		eviction is required (or the request must be split by the caller).
 *
 * 	size_t dset_lru_len - The number of datasets present within the SCC. This
 * 		value is used as the running count needed for DLL debugging and
 *		testing.
 *
 * 	H5SC_stats_t stats - Per-cache statistics used for SCC debugging and
 *		testing. Tracks cache lookups, hits, misses, inserts, removes,
 *		flushes, and evictions. These values are reset when the SCC is
 *		created, destroyed, or reinitialized.
 *
 *
 *	H5SC_dset_header_t *dset_lru_head_ptr – The pointer to the head of the
 *		dataset LRU list (indicating it is the most recently used dataset).
 *		Set to be Null by default or when the dataset LRU list is empty.
 *
 *	H5SC_dset_header_t *dset_lru_tail_ptr – The pointer to the tail of the
 *		dataset LRU list (indicating it is the least recently used dataset).
 *		Set to be Null by default or when the dataset LRU list is empty.
 *
 *	H5SC_chunk_t *chunk_hash_table_head_ptr – The pointer to the head of the SCC
 *		chunk hash table. Set to Null when the hash table is empty or by
 *		default. This pointer will only change when the first element of this
 *		hash table is added or removed.
 *
 *	H5SC_dset_header_t *dset_hash_table_head_ptr – The pointer to the head of
 *		the SCC dataset hash table. Set to Null when this hash table is empty
 *		or by default. This pointer will only change when the first element of
 *		this hash table is added or removed.
 *
 ******************************************************************************/

struct H5SC_stats_t {
    uint64_t scc_lookups;
    uint64_t scc_hits;
    uint64_t scc_misses;

    uint64_t scc_inserts;
    uint64_t scc_removes;

    uint64_t scc_reads_batch_calls_count;
    uint64_t scc_reads_last_batch_len;
    uint64_t scc_reads_max_batch_len;

    uint64_t scc_writes_batch_calls_count;
    uint64_t scc_writes_last_batch_len;
    uint64_t scc_writes_max_batch_len;

    uint64_t scc_chunk_flush_count;
    uint64_t scc_dataset_flush_count;
    uint64_t scc_evictions;
};

/******************************************************************************
 *
 * Structure: H5SC_t
 *
 * Info
 *
 * 	The primary structure used for shared chunk cache management. Provides
 *		fields for maintaining access to the head and tail of the dataset LRU
 *		list, tracking the active and quiescent cache memory usage, and the
 *		pointers necessary to access the hash tables used for chunk and
 *		dataset lookups.
 *
 * Fields
 *
 *	size_t SCC_quiescent_size – The amount of quiescent data, in bytes, present
 *		within the cache when there are no I/O operation being performed on
 *		cached data. This field is computed using the sum of the
 *		curr_dset_size field from each dataset currently in the SCC. This
 *		field is updated whenever a chunk is added, removed, or resized while
 *		present within the SCC. A value of 0 indicates the cache is empty. A
 *		value of 0 is set by default when this structure is created. This value
 *		is also used as the size necessary for DLL operations.
 *
 *	size_t SCC_active_size – The amount of active data, in bytes, allocated to
 *		the SCC when an I/O request is being processed. This field is an
 *		integer-multiple of the current SCC_quiescent_size. The multiplier is
 *		user configurable and is set to be 2x the size of the quiescent limit
 *		by default to accommodate the application of filters or other
 *		operations that often cause chunks to exceed estimated sizes. Once an
 *		I/O request is completed, this field is set to 0.
 *
 *  size_t SCC_max_bytes - The hard cap, in bytes, for resident cache size.
 * 		When SCC_quiescent_size + new_allocation would exceed SCC_max_bytes,
 * 		eviction is required (or the request must be split by the caller).
 *
 * size_t dset_lru_len - The number of datasets present within the SCC. This
 * 		value is used as the running count needed for DLL operations.
 *
 * H5SC_stats_t stats
 *          Per-cache statistics used for SCC debugging and testing. Tracks
 *          cache lookups, hits, misses, inserts, removes, flushes, and
 *          evictions. These values are reset when the SCC is created and may
 *          be reset again when the SCC is destroyed or reinitialized.
 *
 *	H5SC_dset_header_t *dset_lru_head_ptr – The pointer to the head of the
 *		dataset LRU list (indicating it is the most recently used dataset).
 *		Set to be Null by default, when the dataset LRU list is empty, or when
 *		the dataset LRU list contains a single element.
 *
 *	H5SC_dset_header_t *dset_lru_tail_ptr – The pointer to the tail of the
 *		dataset LRU list (indicating it is the least recently used dataset).
 *		Set to be Null by default, when the dataset LRU list is empty, or when
 *		the dataset LRU list contains a single element.
 *
 *	H5SC_chunk_t *chunk_hash_table_head_ptr – The pointer to the head of the SCC
 *		chunk hash table. Set to Null when the hash table is empty or by
 *		default. This pointer will only change when the first element of this
 *		hash table is added or removed.
 *
 *	H5SC_dset_header_t *dset_hash_table_head_ptr – The pointer to the head of
 *		the SCC dataset hash table. Set to Null when this hash table is empty
 *		or by default. This pointer will only change when the first element of
 *		this hash table is added or removed.
 *
 ******************************************************************************/

struct H5SC_t {
    size_t              SCC_quiescent_size;
    size_t              SCC_quiescent_limit;
    size_t              SCC_active_size;
    size_t              SCC_active_limit;
    size_t              dset_lru_len;
    H5SC_stats_t        stats;
    H5SC_dset_header_t *dset_lru_head_ptr;
    H5SC_dset_header_t *dset_lru_tail_ptr;
    H5SC_chunk_t       *chunk_hash_table_head_ptr;
    H5SC_dset_header_t *dset_hash_table_head_ptr;
};

/*****************************/
/*  DLL Structs (some tmp)   */
/*****************************/

/******************************************************************************
 *
 * Structure: H5SC_exhausted_list_t
 *
 * Info
 *
 *   The structure maintains a temporary doubly-linked list of dataset headers
 *   that have been fully processed during a single eviction pass within the
 *   Shared Chunk Cache (SCC). A dataset is added to this list once it can no
 *   longer contribute additional eviction candidates under the current policy
 *   (i.e., all eligible clean and dirty chunks have been considered or the
 *   dataset has reached its minimum reserved size).
 *
 *   This list is used to prevent repeated scanning of the same dataset within
 *   a single eviction pass. After the eviction pass completes, all dataset
 *   headers stored in this list are restored to the global dataset LRU in a
 *   manner that preserves their relative ordering.
 *
 * Fields
 *
 *   H5SC_dset_header_t *head
 *     Pointer to the most recently added dataset header (MRU) in the exhausted
 *     list. This corresponds to the head of the doubly-linked list.
 *
 *   H5SC_dset_header_t *tail
 *     Pointer to the least recently added dataset header (LRU) in the exhausted
 *     list. This corresponds to the tail of the doubly-linked list.
 *
 *   size_t len
 *     Number of dataset headers currently stored in the exhausted list. This
 *     field is maintained explicitly to support sanity checks and fast emptiness
 *     evaluation.
 *
 *   size_t bytes
 *     Aggregate of curr_dset_size for all dataset headers currently stored in
 *     the exhausted list. This field is maintained as a sidecar accounting
 *     mechanism and may be used for debugging, diagnostics, or future policy
 *     extensions.
 *
 ******************************************************************************/
struct H5SC_exhausted_list_t {
    H5SC_dset_header_t *head;
    H5SC_dset_header_t *tail;
    size_t              len;
    size_t              bytes;
};

/*****************************/
/* Package Private Variables */
/*****************************/

/* Add any consts here. */

/******************************/
/* Package Private Prototypes */
/******************************/

/* Looks like there prototypes are referencing local functions (H5SC__...). */

/*
 * H5SC.c Prototypes
 */

/*
 * DLL Prototypes
 */

/* Generic routines */

/******************************************************************************/
/*                         H5SC: DLL Helpers & APIs                           */
/*                                                                            */
/* Purpose                                                                    */
/*   Provide self-contained doubly-linked list (DLL) utilities and small      */
/*   package-private APIs to operate two initial lists without depending on   */
/*   H5C internals:                                                           */
/*     1) Dataset-header LRU over H5SC_t.{dset_lru_head_ptr,dset_lru_tail_ptr}*/
/*     2) Per-dataset chunk LRU over H5SC_dset_header_t.{lru_head_ptr,tail}   */
/*                                                                            */
/* Design                                                                     */
/*   - Splice macros are AUX-like: they ONLY adjust pointers.                 */
/*     Callers explicitly update len/bytes counters in sidecar structs.       */
/*   - Optional sanity helpers can verify linkage and recompute counters.     */
/*   - APIs return herr_t and are intended for use inside the H5SC package.   */
/******************************************************************************/

#ifndef H5SC_DLL_MACROS_H
#define H5SC_DLL_MACROS_H

/* ------------------------------ */
/* Core splice helpers (AUX-like) */
/* ------------------------------ */

/* Prepend NODE as new head (MRU) */
#define H5SC_DLL_PREPEND(node, next_field, prev_field, head, tail, on_fail)                                  \
    do {                                                                                                     \
        (node)->prev_field = NULL;                                                                           \
        (node)->next_field = (head);                                                                         \
        if (head)                                                                                            \
            (head)->prev_field = (node);                                                                     \
        else                                                                                                 \
            (tail) = (node);                                                                                 \
        (head) = (node);                                                                                     \
    } while (0)

/* Append NODE as new tail (LRU) */
#define H5SC_DLL_APPEND(node, next_field, prev_field, head, tail, on_fail)                                   \
    do {                                                                                                     \
        (node)->next_field = NULL;                                                                           \
        (node)->prev_field = (tail);                                                                         \
        if (tail)                                                                                            \
            (tail)->next_field = (node);                                                                     \
        else                                                                                                 \
            (head) = (node);                                                                                 \
        (tail) = (node);                                                                                     \
    } while (0)

/* Remove NODE from the list whose head/tail are HEAD/TAIL */
#define H5SC_DLL_REMOVE(node, next_field, prev_field, head, tail, on_fail)                                   \
    do {                                                                                                     \
        if ((node)->prev_field)                                                                              \
            (node)->prev_field->next_field = (node)->next_field;                                             \
        else {                                                                                               \
            if ((head) != (node)) {                                                                          \
                on_fail;                                                                                     \
            }                                                                                                \
            (head) = (node)->next_field;                                                                     \
        }                                                                                                    \
        if ((node)->next_field)                                                                              \
            (node)->next_field->prev_field = (node)->prev_field;                                             \
        else {                                                                                               \
            if ((tail) != (node)) {                                                                          \
                on_fail;                                                                                     \
            }                                                                                                \
            (tail) = (node)->prev_field;                                                                     \
        }                                                                                                    \
        (node)->next_field = (node)->prev_field = NULL;                                                      \
    } while (0)

/* ------------------------------- */
/* Size accounting (explicit only) */
/* ------------------------------- */

/* Adjust BYTES for a size change while LEN stays unchanged. */
#define H5SC_DLL_UPDATE_FOR_SIZE_CHANGE(len, bytes, old_sz, new_sz, on_fail)                                 \
    do {                                                                                                     \
        if ((new_sz) >= (old_sz))                                                                            \
            (bytes) += (new_sz) - (old_sz);                                                                  \
        else                                                                                                 \
            (bytes) -= (old_sz) - (new_sz);                                                                  \
        (void)(len);                                                                                         \
    } while (0)

/* ------------------------- */
/* Portable sanity utilities */
/* ------------------------- */

/*
 * H5SC_DLL_CHECK_LINKS
 * Verify forward/backward linkage and head/tail correctness.
 *   iter, prev : caller-declared variables of the node pointer type
 *   head, tail : list endpoints
 *   next_field,
 *   prev_field  : member names for next/prev pointers on the node type
 *   on_fail    : statement(s) to execute on invariant violation
 */
#define H5SC_DLL_CHECK_LINKS(iter, prev, head, tail, next_field, prev_field, on_fail)                        \
    do {                                                                                                     \
        (prev) = NULL;                                                                                       \
        if ((head) == NULL) {                                                                                \
            if ((tail) != NULL) {                                                                            \
                on_fail;                                                                                     \
            }                                                                                                \
        }                                                                                                    \
        else {                                                                                               \
            for ((iter) = (head); (iter) != NULL; (iter) = (iter)->next_field) {                             \
                if ((iter)->prev_field != (prev)) {                                                          \
                    on_fail;                                                                                 \
                }                                                                                            \
                (prev) = (iter);                                                                             \
            }                                                                                                \
            if ((tail) != (prev)) {                                                                          \
                on_fail;                                                                                     \
            }                                                                                                \
        }                                                                                                    \
    } while (0)

/*
 * H5SC_DLL_COUNT_BYTES
 * Recompute (len,bytes) by walking the list.
 *   iter      : caller-declared variable of the node pointer type
 *   head      : list head
 *   next_field : member name for next pointer
 *   size_expr : expression per node (e.g., iter->cached_chunk_size)
 *   out_len,
 * out_bytes   : lvalues to receive recomputed totals
 */
#define H5SC_DLL_COUNT_BYTES(iter, head, next_field, size_expr, out_len, out_bytes)                          \
    do {                                                                                                     \
        size_t _n = 0, _b = 0;                                                                               \
        for ((iter) = (head); (iter) != NULL; (iter) = (iter)->next_field) {                                 \
            _n++;                                                                                            \
            _b += (size_t)(size_expr);                                                                       \
        }                                                                                                    \
        (out_len)   = _n;                                                                                    \
        (out_bytes) = _b;                                                                                    \
    } while (0)

#endif /* H5SC_DLL_MACROS_H */

/* Potential debug helper functions */
/******************************************************************************
 *
 * Function:    H5SC_hdr_lru_dump
 *
 * Purpose
 *   Debug helper to print the dataset-header LRU to a stream. If stream is NULL
 *   and HDF5 debugging is enabled, uses H5DEBUG(X).
 *
 ******************************************************************************/
#if defined(H5SC_ENABLE_DUMPS) && (H5SC_ENABLE_DUMPS + 0)
H5_DLL void H5SC_hdr_lru_dump(const char *tag, const struct H5SC_t *sc, FILE *stream);

/******************************************************************************
 *
 * Function:    H5SC_chunk_lru_dump
 *
 * Purpose
 *   Debug helper to print the per-dataset chunk LRU to a stream. If stream is
 *   NULL and HDF5 debugging is enabled, uses H5DEBUG(X).
 *
 ******************************************************************************/
H5_DLL void H5SC_chunk_lru_dump(const char *tag, const struct H5SC_dset_header_t *hdr, FILE *stream);
#endif
#endif /* H5SCpkg_H */
