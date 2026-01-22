#pragma once

#include "blas_interface.h"
#include "lapack_interface.h"

#include <utility> // std::forward

namespace hplinalg {
namespace detail {

/// Small helper to forward all BLAS/LAPACK style calls through a single
/// wrapper. This keeps the call sites uniform and makes it easier to
/// adjust the numerical backend in one place.
template <typename F, typename... Args>
inline auto call_numerics(F&& f, Args&&... args)
    -> decltype(std::forward<F>(f)(std::forward<Args>(args)...))
{
    return std::forward<F>(f)(std::forward<Args>(args)...);
}

/// Cholesky / LDLT factorization of the pivot block.
/// Delegates to lapack_dpotrf(BlockType&).
template <typename BlockT>
inline void factorize_pivot(BlockT& Rii)
{
    call_numerics(lapack_dpotrf, Rii);
}

/// Triangular solve with the factorized pivot block.
/// Delegates to lapack_dpotrs(XBlock&, const RBlock&).
template <typename XBlock, typename RBlock>
inline void solve_pivot(XBlock& X, const RBlock& Rii)
{
    call_numerics(lapack_dpotrs, X, Rii);
}

/// Symmetric rank-k like update on the upper triangular part:
/// C = alpha * A * B^T + beta * C
/// Delegates to blasx_gemmt(C, alpha, A, B, beta).
template <typename C, typename A, typename B, typename Alpha, typename Beta>
inline void symmetric_update(C& Cblk, Alpha alpha,
                             const A& Ablk, const B& Bblk, Beta beta)
{
    call_numerics(blasx_gemmt, Cblk, alpha, Ablk, Bblk, beta);
}

/// General matrix-matrix multiply with transpose on the first factor:
/// C = alpha * A^T * B + beta * C
/// Delegates to blas_gemm_t(C, alpha, A, B, beta).
template <typename C, typename A, typename B, typename Alpha, typename Beta>
inline void gemm_t(C& Cblk, Alpha alpha,
                   const A& Ablk, const B& Bblk, Beta beta)
{
    call_numerics(blas_gemm_t, Cblk, alpha, Ablk, Bblk, beta);
}

/// General matrix-matrix multiply:
/// C = alpha * A * B + beta * C
/// Delegates to blas_gemm(C, alpha, A, B, beta).
template <typename C, typename A, typename B, typename Alpha, typename Beta>
inline void gemm(C& Cblk, Alpha alpha,
                 const A& Ablk, const B& Bblk, Beta beta)
{
    call_numerics(blas_gemm, Cblk, alpha, Ablk, Bblk, beta);
}

} // namespace detail
} // namespace hplinalg
