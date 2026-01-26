void factorize_block_ldlt(DefaultBlocks&       DBlocks,
                          DefaultBlocks&       UBlocks,
                          DefaultBlocks&       SBlocks,
                          DefaultIndexSets&    nnzIndSets,
                          const DefaultReordering& reorderingData,
                          DefaultValuePtr      vals)
{
    using Traits = factorization_traits<
        DefaultBlocks,
        DefaultIndexSets,
        DefaultReordering,
        DefaultValuePtr>;

    detail::FactorizationKernel<Traits>::run(
        DBlocks, UBlocks, SBlocks,
        nnzIndSets, reorderingData, vals);
}
