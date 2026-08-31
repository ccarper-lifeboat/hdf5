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

/*
 * Enable formatted SCC statistics reporting with:
 *     -DH5SC_ENABLE_STAT_DUMPS=1
 *
 * Enable detailed resident-size estimate accuracy statistics with:
 *     -DH5SC_COLLECT_ESTIMATE_STATS=1
 */

/* Get package's private header */
#include "H5SCprivate.h"
#include "H5VMprivate.h"
#include "H5Dprivate.h"
#include "H5Iprivate.h"
#include "H5private.h" /* Included for uthash access through the H5 wrapper */

/* Magic values for structures */
#define H5SC_MAIN_MAGIC     UINT32_C(0x5343548)  /* SCEH */
#define H5SC_DSET_HDR_MAGIC UINT32_C(0x53434448) /* SCDH */
#define H5SC_CHUNK_MAGIC    UINT32_C(0x53434348) /* SCCH */

/**************************/
/* Package Private Macros */
/**************************/

/****************************/
/* Package Private Typedefs */
/****************************/

/******************************************************************************
 *
 * Structure: H5SC_chunk_key_t
 *
 * Info
 *
 * 	The structure stores a unique 128-bit key for a chunk. The key is derived
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
 *             LSB-first bit interleaving yields 0,1,1,0,1,1 → 01101100…2
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
 *		create an instance of H5SC_chunk_t structure. Expected to be set to
 *		1396917064. This value is used to verify that an instance of a
 *		H5SC_chunk_t structure was created properly.
 *
 *	H5SC_chunk_key_t data_key – The unique key associated with the cached
 *		chunk. It is primarily used for chunk lookups through the chunk-
 *		focused hash table and operations that require internal chunk index
 *		updates.
 *
 *	void *chunk_obj – Pointer to the layout-specific decoded resident chunk
 *		object. For the current structured-chunk implementation, this points
 *		to an H5D_chunk_cache_mem_t.
 *
 * 	void *udata – Pointer to layout-specific callback user data associated with
 *		the decoded chunk representation. The SCC treats this pointer as
 *		opaque and passes it through to the applicable layout callbacks during
 *		lookup, decode/encode, read/write processing, flush, erase, and
 *		eviction operations. Ownership and interpretation of the pointed-to
 *		object are defined by the layout implementation rather than by the SCC
 *		itself. For the current structured-chunk implementation, this pointer
 *		refers to an H5D_chunk_ud_t object. When a decoded chunk becomes
 *		resident in the SCC, the associated udata may be retained with the
 *		H5SC_chunk_t for reuse by subsequent callback operations and is
 *		released as part of layout-specific resident-chunk teardown or
 *		eviction.
 *
 *	unsigned ndims – Number of valid dims defined in the scaled coordinates for
 *		this chunk. The default value is 0, a convenient invalid value.
 *
 *	hsize_t chk_log_coord – The linearized, dataset-relative chunk coordinate,
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
 *  size_t cached_chunk_size – Layout-reported total resident allocation
 *      associated with the decoded chunk representation. The value may
 *      include data storage, decoded selection metadata, and other
 *      layout-owned resident allocations. It is not the encoded on-disk
 *      chunk size.
 *
 *      Changes to this field must be routed through
 *      H5SC__chunk_update_cached_size() while the chunk is tracked by SCC.
 *      The default value is zero.
 *
 *	size_t disk_nbytes – The last known size of the on disk data buffer prior to
 *		applying filters or decoding. It is obtained using the
 *		H5D__struct_chunk_lookup callback. The default value is set to 0.
 *
 *	size_t chunk_counter – Pin count for the chunk. A nonzero value indicates
 *		one or more active SCC request pins and makes the chunk ineligible for
 *		reclamation. Currently used to track the pin count for a chunk. A
 *		chunk is pinned to indicate it is a participant in an I/O request to
 *		prevent premature removal from the SCC. Expected to be 0 when not
 *		selected in an I/O request and 1 until it has been decremented during
 *		a typical I/O operation. When this chunk is in an I/O request, it is
 *		expected to be incremented by H5SC__io_info_init() or
 *		H5SC__erase_io_info_init() during the selection process and
 *		decremented by H5SC_read(), H5SC_write(), or H5SC_erase() (or within
 *		H5SC__invoke_read_dset_batched() or H5SC__invoke_write_dset_batched()
 *		during chunk-by-chunk debugging).
 *
 *	H5SC_tag_t last_op – An enum-typed tag used during debugging to identify the
 *		most recent internal SCC operation performed on this chunk. Set to
 *		H5SC_TAG_CREATE when an H5SC_chunk_t structure is created. See the
 *		section defining H5SC_tag_t for possible values.
 *
 *	bool dirty_flag – A flag to indicate when an in-memory chunk buffer has been
 *		modified. A value of True indicates a chunk buffer is dirty. This flag
 *		is set when a chunk is created/modified/resized. Set to False by
 *		default.
 *
 *	bool partial_IO – Indicates that the resident chunk has participated in a
 *		partial read or write operation, meaning that the chunk-local
 *		selection represented less than the complete logical chunk. This field
 *		records SCC processing state associated with the resident
 *		representation and may be used by operations that need to distinguish
 *		partial-I/O state from a fully processed chunk. The field is not
 *		currently used to rank chunks within the active or quiescent
 *		reclamation policies. Reclamation eligibility and ordering are
 *		determined separately through pin state, resident size, dirty state,
 *		dataset retention policy, and LRU position.
 *
 *		The field is set to false by default.
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
 * Structure: H5SC_io_scratch_t
 *
 * Info
 *
 *      The structure maintains reusable, dynamically sized scratch storage
 *      used while processing SCC read, write, lookup, and related I/O
 *      operations.
 *
 *      Many SCC layout callbacks accept vector arguments whose elements are
 *      pointers to scalar values. Historically, these vectors and their
 *      individual scalar objects were allocated separately for each I/O
 *      invocation. H5SC_io_scratch_t provides reusable pointer vectors and
 *      contiguous backing arrays for those values so that repeated SCC
 *      operations do not require large numbers of small heap allocations.
 *
 *      The scratch storage is associated with an H5SC_io_info_t structure
 *      and therefore with a single SCC dataset header. Capacity is grown as
 *      necessary to accommodate an I/O window and is retained for reuse by
 *      subsequent operations on the same dataset.
 *
 *      Scratch contents are transient and have no semantic meaning between
 *      I/O operations. Callers must initialize the portion of each array that
 *      they intend to use before invoking a layout callback. Pointers stored
 *      in the vector-view fields reference the corresponding contiguous
 *      backing-storage arrays and must not be independently freed.
 *
 *      The scratch structure does not own decoded chunk objects, layout-
 *      specific udata, selected-chunk dataspaces, or user buffers. Pointer
 *      arrays such as udata and chunk may temporarily reference
 *      such objects, but ownership remains governed by the normal SCC/layout
 *      callback contracts.
 *
 * Lifetime
 *
 *      H5SC_io_scratch_t is dataset-owned through H5SC_io_info_t.
 *
 *      - It is allocated lazily when scratch capacity is first required.
 *
 *      - H5SC__io_scratch_ensure() grows the backing arrays when an operation
 *        requires more entries than are currently available. Existing scratch
 *        contents may be discarded when growth occurs.
 *
 *      - H5SC__io_info_reset() does not free or shrink scratch storage.
 *        Resetting an I/O request therefore retains the allocated capacity for
 *        reuse by later operations on the same dataset.
 *
 *      - H5SC__io_info_term() permanently releases the scratch arrays and the
 *        H5SC_io_scratch_t structure when the dataset's persistent I/O state
 *        is destroyed.
 *
 *      Scratch storage must not contain the sole owning reference to an
 *      allocation when H5SC__io_info_reset() or H5SC__io_info_term() is
 *      called. Any callback-generated object requiring persistent ownership
 *      must first be transferred to its appropriate H5SC_chunk_t or otherwise
 *      released.
 *
 * Fields
 *
 *  *** General callback-vector views ***
 *
 *      size_t alloced – Number of entries currently available in each
 *          reusable scratch vector. A value of 0 indicates that no vector
 *          backing storage has been allocated.
 *
 *      const hsize_t **scaled – Reusable vector of pointers to scaled chunk
 *          coordinates. The coordinates themselves are owned elsewhere, normally
 *          by H5SC_io_sel_chunk_t or H5SC_chunk_t.
 *
 *      haddr_t **addr – Callback-facing vector whose entries point into
 *          addr_values.
 *
 *      hsize_t **size – Callback-facing vector whose entries point into
 *          size_values.
 *
 *      hsize_t **defined_values_size – Callback-facing vector whose entries
 *          point into defined_values_size_values.
 *
 *      size_t **size_hint – Callback-facing vector whose entries point into
 *          size_hint_values.
 *
 *      size_t **defined_values_size_hint – Callback-facing vector whose
 *          entries point into defined_values_size_hint_values.
 *
 *      void **udata – Reusable vector for layout-specific callback udata
 *          pointers. The vector does not own the objects referenced by its
 *          elements.
 *
 *  *** Contiguous backing storage for scalar callback outputs ***
 *
 *      haddr_t *addr_values – Contiguous scalar backing storage for addr.
 *
 *      hsize_t *size_values – Contiguous scalar backing storage for size.
 *
 *      hsize_t *defined_values_size_values – Contiguous scalar backing
 *          storage for defined_values_size.
 *
 *      size_t *size_hint_values – Contiguous scalar backing storage for
 *          size_hint.
 *
 *      size_t *defined_values_size_hint_values – Contiguous scalar backing
 *          storage for defined_values_size_hint.
 *
 *  *** Read/write processing state ***
 *
 *      size_t *miss_idx – Reusable vector containing indexes of selected
 *          chunks requiring miss-path processing.
 *
 *      size_t *hit_idx – Reusable vector containing indexes of selected
 *          chunks handled through resident-hit processing.
 *
 *      void **chunk – Reusable vector of chunk-object or temporary chunk-
 *          buffer pointers. Ownership is determined by the processing path and
 *          is not implied by storage in this vector.
 *
 *  *** Lookup-miss batching ***
 *
 *      const hsize_t **lookup_scaled – Scratch vector of scaled coordinates
 *          used when batching layout lookup misses.
 *
 *      haddr_t **lookup_addr – Callback-facing address vector used by
 *          miss-only lookup batching.
 *
 *      hsize_t **lookup_size – Callback-facing encoded-size vector used by
 *          miss-only lookup batching.
 *
 *      hsize_t **lookup_defined_values_size – Callback-facing defined-value
 *          metadata-size vector used by miss-only lookup batching.
 *
 *      size_t **lookup_size_hint – Callback-facing encoded-buffer allocation
 *          hint vector used by miss-only lookup batching. For the current
 *          structured-chunk implementation, each nonzero value equals the
 *          encoded on-disk chunk size and is used to allocate the raw buffer
 *          passed to decode.
 *
 *      size_t **lookup_defined_values_size_hint – Callback-facing allocation
 *          hint vector for the encoded defined-values metadata buffer. Values
 *          are byte sizes, not defined-element counts.
 *
 *      void **lookup_udata – Temporary vector receiving layout-specific udata
 *          from miss-only lookup batching. The vector itself is reusable; the
 *          pointed-to objects are not owned by H5SC_io_scratch_t.
 *
 *      size_t *lookup_idx – Reusable mapping from entries in the compact
 *          miss-only lookup vectors back to indexes in the active
 *          H5SC_io_sel_chunk_t selection window.
 *
 *      size_t *order_idx – Reusable vector containing absolute indexes into
 *          the owning H5SC_io_info_t::sel_chunks allocation. For read and write
 *          requests, the valid prefix is populated in stable resident-first
 *          order: selections whose cached chunks already contain decoded resident
 *          objects appear first, followed by selections requiring lookup or
 *          materialization. Relative order within each group is preserved.
 *
 *          The vector is owned by H5SC_io_scratch_t, allocated and grown by
 *          H5SC__io_scratch_ensure(), retained across H5SC__io_info_reset(), and
 *          released by H5SC__io_info_term(). Entries have no semantic validity
 *          between I/O requests.
 *
 ******************************************************************************/
struct H5SC_io_scratch_t {
    size_t alloced;

    const hsize_t **scaled;
    haddr_t       **addr;
    hsize_t       **size;
    hsize_t       **defined_values_size;
    size_t        **size_hint;
    size_t        **defined_values_size_hint;
    void          **udata;

    haddr_t *addr_values;
    hsize_t *size_values;
    hsize_t *defined_values_size_values;
    size_t  *size_hint_values;
    size_t  *defined_values_size_hint_values;

    size_t *miss_idx;
    size_t *hit_idx;
    void  **chunk;

    const hsize_t **lookup_scaled;
    haddr_t       **lookup_addr;
    hsize_t       **lookup_size;
    hsize_t       **lookup_defined_values_size;
    size_t        **lookup_size_hint;
    size_t        **lookup_defined_values_size_hint;
    void          **lookup_udata;
    size_t         *lookup_idx;

    size_t *order_idx;
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
 *	  an I/O operation, the starting position (used for batching), how many chunks
 *	  are present in the request, and how large sel_chunks currently is (in
 *	  bytes). These fields are subsequently consumed by I/O execution paths and
 *	  chunk processing routines. Generally speaking, the following relations are
 *    expected to be true for the chunks selected for this dataset:
 *  The following relations are expected for an active request:
 *
 *      * 0 <= num_resident_sel_chunks <= num_sel_chunks
 *      * num_sel_chunks <= sel_chunks_alloced
 *      * if sel_order is NULL:
 *            sel_start + num_sel_chunks <= sel_chunks_alloced
 *      * if sel_order is non-NULL:
 *            sel_order[j] < sel_chunks_alloced for every active j
 *
 * Fields
 *
 * 	H5SC_io_sel_chunk_t *sel_chunks – Pointer to a vector of structures
 *		describing the chunks selected for the current I/O operation. Each
 *		element contains per-chunk selection and metadata required to
 *		execute I/O on that chunk. Default size of 128; maximum size is defined
 *      by sel_chunks_alloced. The number of non-NULL elements present is
 *      defined by num_sel_chunks.
 *
 *  const size_t *sel_order – Non-owning view of the ordering vector used
 *      to resolve the active selected-chunk window. When non-NULL,
 *      sel_order[j] is the absolute index of relative selection j in the
 *      stable sel_chunks allocation. When NULL, selections are resolved
 *      contiguously as sel_chunks[sel_start + j].
 *
 *      During normal read and write initialization, this field references
 *      scratch->order_idx. Invoke-layer batching may temporarily advance the
 *      pointer to expose a subrange of that order vector. The pointer must
 *      never be freed independently and must be restored after batched
 *      processing.
 *
 *  H5SC_io_scratch_t *scratch – Pointer to reusable dataset-owned scratch
 *      storage used by SCC I/O processing. The structure is allocated lazily
 *      and its capacity is retained across H5SC__io_info_reset() calls.
 *      H5SC__io_info_term() permanently releases this storage.
 *
 *	size_t sel_start – The starting index within sel_chunks for processing.
 *		This field is typically used to support batched processing of selected
 *		chunks.
 *
 *	size_t num_sel_chunks – The number of valid chunk entries currently
 *		stored in sel_chunks. Serves as a count for the number of chunks
 *		participating in the I/O request.
 *
 *  size_t num_resident_sel_chunks – Number of resident decoded chunk
 *      selections at the beginning of the complete resident-first order
 *      vector. A selection is resident when its associated H5SC_chunk_t has a
 *      non-NULL chunk_obj.
 *
 *      The invoke-layer batching functions use this count as the boundary
 *      between the resident and nonresident phases so that a batch never
 *      crosses from resident processing into lookup/materialization
 *      processing. This count describes the complete request order, not the
 *      length of an individual active batch.
 *
 *	size_t sel_chunks_alloced – The total number of elements allocated for the
 *     sel_chunks vector. This value defines the current maximum capacity of
 *     the vector and is used to determine when reallocation is required to
 *     accommodate more chunks. This value is 128 by default and increases by
 *     multiples of 2 as necessary.
 *
 ******************************************************************************/
struct H5SC_io_info_t {
    H5SC_io_sel_chunk_t *sel_chunks;
    const size_t        *sel_order;
    H5SC_io_scratch_t   *scratch;
    size_t               sel_start;
    size_t               num_sel_chunks;
    size_t               num_resident_sel_chunks;
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
 *		chunk’s offset in each dimension and are given from the upper left
 *		hand corner.
 *
 *	hsize_t scaled[H5S_MAX_RANK] – The scaled chunk coordinates, computed
 *		by dividing coords by the chunk dimensions for each rank. These
 *		values are used for chunk indexing, hashing, and cache lookup
 *		operations.
 *
 *	hsize_t chk_log_coord – The linearized, dataset-relative chunk coordinate,
 *		referred to as the chunk logical coordinate (index). This value is
 *		computed from the dataset dimensions, chunk dimensions, and element
 *		position. It is computed when a H5SC_chunk_t structure is added to the
 *		SCC and is used as one of the components for the chunk key
 *		(H5SC_chunk_key_t). This value is stored as a reference for checking
 *		whether a chunk should be reindexed when a dataset contains one or
 *		more extensible dimensions.
 *
 *	H5SC_chunk_t *cached_chunk – Pointer to the corresponding SCC cache
 *		entry for this chunk. This is populated when the chunk is found
 *		or inserted into the cache and is NULL if no cache entry has
 *		been associated yet.
 *
 *  bool pin_held – Indicates that this selection entry owns one increment
 *      of cached_chunk->chunk_counter. The flag is set immediately after the
 *      selection initialization path increments the chunk counter and is
 *      cleared immediately after the corresponding read, write, erase, or
 *      error-cleanup path decrements it.
 *
 *      This field tracks per-selection ownership and must be used when
 *      releasing pins; chunk_counter > 0 alone does not prove that a
 *      particular selection owns a pin. A selection must not decrement the
 *      chunk counter when pin_held is false.
 *
 *	bool lookup_valid – Indicates whether the lookup-related fields
 *		contain valid information for this chunk.
 *
 *	haddr_t lookup_addr – The on-disk address obtained from lookup
 *		operations.
 *
 *  size_t lookup_disk_nbytes – Last known encoded on-disk size of the structured
 *      chunk, after application of any applicable filter pipelines. The value
 *      is obtained from layout lookup or updated after insertion. It does not
 *      describe the decoded resident allocation. The default value is zero.
 *
 *  hsize_t lookup_defined_values_size – Encoded byte size occupied by
 *      defined-values selection metadata returned by lookup. This is a byte
 *      count, not a count of defined elements.
 *
 *  size_t lookup_size_hint – Layout-provided allocation hint for the encoded
 *      chunk buffer used by the raw read/decode path. For the current
 *      structured-chunk layout implementation, this value is the encoded
 *      on-disk chunk size returned by lookup.
 *
 *      This value is a byte size, not a defined-value count, and does not
 *      describe the complete decoded resident allocation. It must not be used
 *      as H5SC_SIZE_EST_LOOKUP unless a future layout callback provides a
 *      separately documented complete resident-allocation estimate.
 *
 *  size_t lookup_defined_values_size_hint – Layout-provided allocation hint
 *      for reading the encoded defined-values selection metadata. For the
 *      current structured-chunk implementation, this value equals
 *      lookup_defined_values_size.
 *
 *      It is an encoded byte size, not a count of defined elements and not a
 *      complete decoded resident-allocation estimate.
 *
 *	void *lookup_udata – Pointer to user data structure used during
 *		lookup callbacks.
 *
 *  size_t estimate_resident_size – Predicted final resident allocation size
 *      for this selected chunk. The invoke-layer batching code sets this
 *      before admitting a nonresident chunk into a batch. The value remains
 *      valid only while estimate_pending is true and is reset when the
 *      estimate is recorded or the request state is reset.
 *
 *  H5SC_size_est_source_t estimate_source – Identifies the source used to
 *		produce estimate_resident_size. In the current implementation,
 *		estimates are produced from dataset-local resident-size history when
 *		available and otherwise from the dense logical chunk size.
 *		H5SC_SIZE_EST_LOOKUP and H5SC_SIZE_EST_SPARSE_MODEL are reserved for
 *		future estimation mechanisms and are not currently selected by
 *		H5SC__estimate_resident_size(). Set to H5SC_SIZE_EST_NONE when no
 *		estimate is pending.
 *
 *  bool estimate_pending – Indicates that estimate_resident_size and
 *      estimate_source describe a materialization estimate that has not yet
 *      been compared with an actual resident allocation. A successful read
 *      materialization or write gather records the estimate and clears this
 *      flag. Failed or abandoned requests clear it during request reset
 *      without recording an observation.
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
 *	bool file_space_shared – Ownership flag for file_space. A value of false
 *		indicates that the dataspace is a temporary per-chunk selection owned
 *		by the SCC and must be closed by SCC request cleanup. A value of true
 *		indicates that file_space is a borrowed reference to a dataspace owned
 *		elsewhere; SCC may use the dataspace for the current selection record
 *		but must not close it. The current read/write initialization path
 *		constructs an SCC-owned file_space for each selected chunk, so this
 *		flag normally remains false. The flag exists to distinguish owned
 *		temporary dataspaces from borrowed dataspace references if such reuse
 *		is employed by a processing path.
 *
 *	bool mem_space_shared – Ownership flag for mem_space. A value of false
 *		indicates that the dataspace is a temporary per-chunk memory selection
 *		created and owned by the SCC and must be closed by SCC request
 *		cleanup. A value of true indicates that mem_space is a borrowed
 *		reference to a dataspace owned elsewhere and must not be closed by the
 *		SCC. In the current implementation, this occurs when an I/O request
 *		selects exactly one chunk: the selected-chunk record directly
 *		references dset_info->mem_space rather than creating a separate per-
 *		chunk memory dataspace.
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
    bool                    pin_held;
    haddr_t                 lookup_addr;
    hsize_t                 lookup_disk_nbytes;
    hsize_t                 lookup_defined_values_size;
    size_t                  lookup_size_hint;
    size_t                  lookup_defined_values_size_hint;
    void                   *lookup_udata;
    size_t                  estimate_resident_size;
    H5SC_size_est_source_t  estimate_source;
    bool                    estimate_pending;
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
 *		created/destroyed consistently.
  *
 * 	haddr_t dset_addr –  The 64-bit address of the object header for this
 *		dataset. This field is used as the key for the dataset hash table.
 *		It is set to 0 by default.
 *
 *  min_dset_size – Preferred minimum resident byte count retained for this
 *      dataset while trimming SCC to its quiescent limit. It is a steady-state
 *      retention target, not a hard reservation. Active-pressure eviction may
 *      reduce the dataset below this value when necessary to satisfy
 *      SCC_active_limit.
 *
 * 	size_t chunk_lru_len – The number of chunks present within the SCC. This
 * 		value is used as the running count needed for DLL operations. Set to 0
 *		by default.
 *
 *	size_t curr_dset_size – The current amount of chunk data, in bytes
 *		associated with this dataset within the cache (i.e., is a sum of
 *		current chunk sizes). This quantity is updated when a chunk is added,
 *		removed, or resized. This field contributes to total cache
 *		utilization tracked in H5SC_t. It is set to 0 by default.
 *
 *  size_t resident_estimate_high – Decaying high-water observation of final
 *		resident chunk allocation sizes for this dataset. Used together with
 *		resident_estimate_ema to produce the dataset-history estimate after
 *		one or more successful resident-size observations have been recorded.
 *		Updated only after successful final resident-size accounting.
 *
 *  size_t resident_estimate_ema – Exponential moving average of observed final
 *		resident chunk allocation sizes for this dataset. Used together with
 *		resident_estimate_high to form the current dataset-history estimate
 *		when resident-size observations are available.
 *
 *  uint64_t resident_estimate_samples – Number of resident allocation
 *      observations incorporated into the dataset-local adaptive estimate.
 *      Saturates at UINT64_MAX.
 *
 *  size_t reclaimable_clean_bytes – Number of resident bytes currently held
 *      by unpinned, nonzero, clean chunks in this dataset that may be evicted
 *      under active-pressure policy. This value does not apply
 *      min_dset_size; that field is only a quiescent-retention target.
 *
 *  size_t reclaimable_dirty_bytes – Number of resident bytes currently held
 *      by unpinned, nonzero, dirty chunks in this dataset that may be reclaimed
 *      by flush-and-evict under active-pressure policy. This value does not
 *      apply min_dset_size.
 *
 *	H5SC_tag_t last_op – An enum-typed tag used during debugging to identify the
 *		most recent internal SCC operation performed on this dataset header.
 *		Tags representing operations common to chunks and dataset headers are
 *		interpreted in the context of the structure containing the tag. See
 *		the section defining H5SC_tag_t for possible values.
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
 *	H5D_t *dset – Pointer to the dataset; included for debugging convenience.
 *
 *	H5SC_chunk_t *lru_head_ptr – The pointer to the head of this dataset’s chunk
 *		LRU list (indicating it is the most recently used chunk). Set to
 *		Null by default or when the chunk LRU list is empty.
 *
 *	H5SC_chunk_t *lru_tail_ptr – The pointer to the tail of this dataset’s chunk
 *		LRU list (indicating it is the least recently used chunk). Set to Null
 *		by default or when the chunk LRU list is empty.
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
    size_t                     resident_estimate_high;
    size_t                     resident_estimate_ema;
    uint64_t                   resident_estimate_samples;
    size_t                     reclaimable_clean_bytes;
    size_t                     reclaimable_dirty_bytes;
    H5SC_tag_t                 last_op;
    H5SC_io_info_t            *io_info;
    bool                       resize_in_progress;
    struct H5D_t              *dset;
    struct H5SC_chunk_t       *lru_head_ptr;
    struct H5SC_chunk_t       *lru_tail_ptr;
    struct H5SC_dset_header_t *next_dset_ptr;
    struct H5SC_dset_header_t *prev_dset_ptr;
    UT_hash_handle             dset_hval;
};

/******************************************************************************
 *
 * Structure: H5SC_stats_t
 *
 * Purpose
 *      Maintains debugging and test statistics for a single SCC instance.
 *      These counters are used to observe cache behavior, particularly the
 *      rate at which chunk lookups resolve to resident non-placeholder
 *      chunks already present in the SCC.
 *
 * Fields
 *
 *	uint64_t scc_lookups - The total number of SCC chunk lookup operations
 *		recorded for this cache instance.
 *
 *	uint64_t scc_hits - The number of SCC chunk lookups that resolved to a
 *		resident non-placeholder chunk already present within the cache.
 *
 *	uint64_t scc_misses - The number of SCC chunk lookups that did not resolve
 *		to a resident non-placeholder chunk already present within the cache.
 *
 *	uint64_t scc_inserts - The number of chunk insert operations recorded for
 *		this cache instance.
 *
 *	uint64_t scc_removes - The number of chunk remove operations recorded for
 *		this cache instance.
 *
 *	uint64_t scc_chunk_flush_count -  The number of chunk flush operations
 *		recorded for this cache instance.
 *
 *	uint64_t scc_dataset_flush_count - The number of dataset flush operations
 *		recorded for this cache instance.
 *
 *	uint64_t scc_evictions - The number of chunk or dataset eviction operations
 *		recorded for this cache instance.
 *
 *	uint64_t scc_reads_batch_calls_count - The number of individual read batches
 *		recorded for this cache instance.
 *
 *	uint64_t scc_reads_last_batch_len - The number of chunks processed in the
 *		last processed read batch for this cache instance.
 *
 *	uint64_t scc_reads_max_batch_len - The largest number of chunks processed in
 *		a single read batch for this cache instance.
 *
 *	uint64_t scc_writes_batch_calls_count - The number of individual write
 *		batches recorded for this cache instance.
 *
 *	uint64_t scc_writes_last_batch_len - The number of chunks processed in the
 *		last processed write batch for this cache instance.
 *
 *	uint64_t scc_writes_max_batch_len - The largest number of chunks processed
 *		in a single write batch for this cache instance.
 *
 *	uint64_t scc_oversized_admission_count – Number of indivisible chunks
 *		admitted through the oversized single-chunk exception.
 *
 *	size_t scc_oversized_admission_max_bytes – Largest incremental resident-
 *		size estimate admitted through the exception.
 *
 *	size_t scc_oversized_admission_max_excess – Largest amount by which an
 *		oversized estimate exceeded SCC_active_limit. This may be zero when
 *		the exception was triggered by reduced effective budget rather than by
 *		an estimate larger than the configured limit.
 *
 *	uint64_t size_estimate_count - Number of pending resident-size estimates
 *		compared with a successfully materialized final resident allocation.
 *
 *	uint64_t size_estimate_under_count - Number of comparisons for which the
 *		estimate was less than the final resident allocation.
 *
 *	uint64_t size_estimate_over_count - Number of comparisons for which the
 *		estimate was greater than the final resident allocation. Exact
 *		estimates are included only in size_estimate_count.
 *
 *	size_t size_estimated_total – Saturating sum of all compared resident-size
 *		estimates.
 *
 *	size_t size_actual_total – Saturating sum of the corresponding final
 *		resident allocations.
 *
 *	size_t size_overestimated_bytes – Saturating sum of estimate - actual for
 *		overestimates.
 *
 *	size_t size_underestimated_bytes – Saturating sum of actual - estimate for
 *		underestimates.
 *
 *	size_t size_estimate_max_over – Largest observed estimate – actual
 *		difference.
 *
 *	size_t size_estimate_max_under – Largest observed actual – estimate
 *		difference.
 *
 *	uint64_t size_estimate_*_count – Number of recorded comparisons attributed
 *		to each estimate source. The sum of the source counts should equal
 *		size_estimate_count.
 *
 *	uint64_t write_growth_count – Number of materialized writes for which
 *		final resident size growth was measured.
 *
 *	size_t write_growth_total – Saturating sum of measured write-growth bytes.
 *
 *	size_t write_growth_max – Largest measured write-growth value.
 *
 ******************************************************************************/

struct H5SC_stats_t {
    uint64_t scc_lookups;
    uint64_t scc_hits;
    uint64_t scc_misses;

    uint64_t scc_inserts;
    uint64_t scc_removes;

    uint64_t scc_chunk_flush_count;
    uint64_t scc_dataset_flush_count;
    uint64_t scc_evictions;

    uint64_t scc_reads_batch_calls_count;
    uint64_t scc_reads_last_batch_len;
    uint64_t scc_reads_max_batch_len;

    uint64_t scc_writes_batch_calls_count;
    uint64_t scc_writes_last_batch_len;
    uint64_t scc_writes_max_batch_len;

    uint64_t scc_oversized_admission_count;
    size_t   scc_oversized_admission_max_bytes;
    size_t   scc_oversized_admission_max_excess;

#if defined(H5SC_COLLECT_ESTIMATE_STATS) && (H5SC_COLLECT_ESTIMATE_STATS + 0)
    /* General Size Estimate Stats*/
    uint64_t size_estimate_count;
    uint64_t size_estimate_under_count;
    uint64_t size_estimate_over_count;

    size_t size_estimated_total;
    size_t size_actual_total;
    size_t size_overestimated_bytes;
    size_t size_underestimated_bytes;

    size_t size_estimate_max_over;
    size_t size_estimate_max_under;

    /* Specific Estimate Source Stats */
    uint64_t size_estimate_lookup_count;
    uint64_t size_estimate_sparse_model_count;
    uint64_t size_estimate_dense_fallback_count;
    uint64_t size_estimate_dataset_history_count;

    /* Write Specific Estimate Stats*/
    uint64_t write_growth_count;
    size_t   write_growth_total;
    size_t   write_growth_max;
#endif
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
 *	uint64_t SCC_magic – Value set when a H5SC_t structure is created.
 *		This value is used for debugging to ensure that the structure is
 *		created/destroyed consistently.
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
 *  size_t SCC_active_size – Reserved accounting field for transient SCC
 *      allocations that are not included in SCC_quiescent_size. Current chunk
 *      resident-size accounting is maintained through SCC_quiescent_size, and
 *      admission uses SCC_active_limit together with cached active-policy
 *      reclaimability. This field is not currently part of that admission
 *      calculation and remains reserved for future transient-memory
 *      accounting.
 *
 *	size_t SCC_active_limit – Normal operational ceiling used when admitting
 *		additional resident chunk bytes during an I/O request. If current
 *		resident bytes plus the requested incremental allocation exceed this
 *		limit, H5SC__ensure_space() performs active-pressure reclamation.
 *		Active-pressure reclamation may reduce a dataset below min_dset_size.
 *
 *		If one indivisible chunk cannot fit within the effective active
 *		budget, the SCC may temporarily exceed this limit while processing
 *		that chunk. The configured limit is not modified, and ordinary active-
 *		limit enforcement is restored immediately after the oversized chunk is
 *		unpinned. The value is configured by max_a_size.
 *
 *		SCC_active_limit is the normal operational ceiling rather than an
 *		absolute prohibition against processing an individual chunk larger
 *		than the available active budget. When a single indivisible chunk
 *		cannot otherwise be admitted, SCC may temporarily exceed this limit
 *		through the oversized single-chunk admission path. The configured
 *		value is not modified by this exception; normal active-limit
 *		enforcement resumes after the oversized chunk is processed and
 *		unpinned.
 *
 * 	size_t dset_lru_len - The number of datasets present within the SCC. This
 * 		value is used as the running count needed for DLL debugging and
 *		testing.
 *
 *  size_t reclaimable_clean_bytes – Cache-wide sum of dataset-local clean
 *      bytes reclaimable under active-pressure policy. The value includes all
 *      unpinned, nonzero, clean resident chunks and does not enforce
 *      dataset-local min_dset_size.
 *
 *  size_t reclaimable_dirty_bytes – Cache-wide sum of dataset-local dirty
 *      bytes reclaimable through flush-and-evict under active-pressure policy.
 *      The value includes all unpinned, nonzero, dirty resident chunks and
 *      does not enforce dataset-local min_dset_size.
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
 * #if H5SC_DO_SANITY_CHECKS
 *  bool test_fail_next_chunk_flush - Test-only one-shot injection flag.
 *      When true, the next attempt to flush a dirty chunk fails before
 *      encoding or storage metadata changes occur. The flush path resets the
 *      field to false when consuming the injected failure.
 * #endif
 *
 ******************************************************************************/

struct H5SC_t {
    uint64_t            SCC_magic;
    size_t              SCC_quiescent_size;
    size_t              SCC_quiescent_limit;
    size_t              SCC_active_size;
    size_t              SCC_active_limit;
    size_t              dset_lru_len;
    size_t              reclaimable_clean_bytes;
    size_t              reclaimable_dirty_bytes;
    H5SC_stats_t        stats;
    H5SC_dset_header_t *dset_lru_head_ptr;
    H5SC_dset_header_t *dset_lru_tail_ptr;
    H5SC_chunk_t       *chunk_hash_table_head_ptr;
    H5SC_dset_header_t *dset_hash_table_head_ptr;

#if H5SC_DO_SANITY_CHECKS
    /*
     * Test-only one-shot failure injection. When true, the next dirty-chunk
     * flush fails before encoding begins and resets this field to false.
     */
    bool test_fail_next_chunk_flush;
#endif
};

/*****************************/
/*  DLL Structs (some tmp)   */
/*****************************/

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
#if defined(H5SC_ENABLE_STAT_DUMPS) && (H5SC_ENABLE_STAT_DUMPS + 0)
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
