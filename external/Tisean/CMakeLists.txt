set ( tisean_SOURCES check_alloc.c
check_option.c
eigen.c
exclude_interval.c
find_multi_neighbors.c
find_neighbors.c
get_multi_series.c
get_series.c
invert_matrix.c
make_box.c
make_multi_box2.c
make_multi_box.c
make_multi_index.c
myfgets.c
rand_arb_dist.c
rand.c
rescale_data.c
scan_help.c
search_datafile.c
solvele.c
test_outfile.c
variance.c
what_i_do.c )

add_library( ddtsa ${tisean_SOURCES} )
