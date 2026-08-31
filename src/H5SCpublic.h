/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by Lifeboat, LLC                                                *
 * All rights reserved.                                                      *
 *                                                                           *
 * The full copyright notice, including terms governing use, modification,   *
 * and redistribution, is contained in the COPYING file, which can be found  *
 * at the root of the source code distribution tree.                         *
 * If you do not have access to either file, you may request a copy from     *
 * help@lifeboat.llc                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*-------------------------------------------------------------------------
 *
 * Created:     H5SCpublic.h
 *
 * Purpose:     Public header file for shared chunk cache functions
 *
 *-------------------------------------------------------------------------
 */
#ifndef H5SCpublic_H
#define H5SCpublic_H
#if 0
#include "H5public.h" /* Generic Functions                        */
#endif
/* notes:  structure will need doxigen annotations everually */

/*******************************************************************************
 *
 * struct H5SC__cache_config_t
 *
 * The H5SC__cache_config_t structure is used to aggregate shared chunk cache
 * configuration data.  The fields in the structure are
 * discussed individually below.
 *
 * version:     int field containing the version number of this version
 *              of the H5SC__cache_config_t structure.  Any instance of
 *              H5SC__scc_config_t passed to the cache must have a known
 *              version number, or an error will be flagged.
 *
 * max_q_size:  size_t field containing the maximum number of bytes that
 *              the shared chunk cache may use while quiescent -- that is
 *              when not actively engaged in serving an I/O request
 *
 * max_a_size:  size_t field containing the normal maximum number of resident
 *		        bytes that the shared chunk cache may use while serving an I/O
 *		        request. If one indivisible chunk cannot be processed within this
 *		        limit, the SCC may temporarily exceed max_a_size for the duration
 *		        of that single-chunk operation. Before admitting the chunk, the
 *		        SCC reclaims eligible resident data under the active-pressure
 *		        policy and returns to normal configured-limit enforcement after
 * 		        the chunk is processed and unpinned. Applications using large
 *		        logical chunks should configure an active limit sufficient for the
 *		        expected working set.
 *
 *              If one indivisible chunk cannot be processed within this
 *              limit, the SCC may temporarily exceed max_a_size for the
 *              duration of that single-chunk operation. The SCC reclaims
 *              eligible resident data before admitting the chunk and returns
 *              to normal configured-limit enforcement after processing it.
 *              Applications using large logical chunks should configure an
 *              active limit sufficient for the expected working set.
 *
 ******************************************************************************/

#define H5SC__CURR_SCC_VERSION 0

typedef struct H5SC__cache_config_t {

    int version;

    size_t max_q_size;

    size_t max_a_size;

} H5SC__cache_config_t;

#endif
