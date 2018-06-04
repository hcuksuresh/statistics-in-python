RandomForrest

RF is biased towards the continuous features. If you have ordinal or categorical variables, they are less likely to be selected for the splits. This is a known problem for RF. 

Random forest can sometimes be tricky. As you mentioned you have sparse features, so majority of them are zeroes. So randomly when you choose features for splitting the node , zero features may be selected , which may be one of the reasons for high error rate.

One method you could do is to identify informative features and then use only those features to do splits at the node. This should help reduce error rate.

https://www.researchgate.net/post/Im_trying_to_apply_random_forests_in_a_sparse_data_set_Unfortunately_there_is_more_than_40_error_in_my_result_Can_anyone_suggest_where_to_refine


