__author__ = 'redhat'

# preprocess.py
# def make_dict_of_elem(list_of_tuples):
#     """
#     Assume that we have a list of tuples.
#     An element in n-th coordinate of a tuple comes from a set.
#     The set contains all elements that are in n-th coordinate of all
#     tuples in the list.
#     Map all the elements from the set to a nonnegative integer and vice versa.
#     Returns 3 lists.
#     The first list contains dictionaries of k-th element to nonnegative integer
#     The second is the other way around.
#     the third contains size of each set
#     """
#
#     # initialize list of dictionaries
#     list_to_int = []
#     list_to_elem = []
#     list_size = []
#
#     # assume that all tuples have same length
#     for i in range(len(list_of_tuples[0])):
#         list_to_int.append({})  # key: elem, value: int
#         list_to_elem.append({})  # key: int, value: elem
#         counter = 0
#         for tuple in list_of_tuples:
#             # unpack elements in the tuple
#             # store it in dictionary if the key doesn't exist, pass otherwise
#             if tuple[i] not in list_to_int[i]:
#                 list_to_int[i][tuple[i]] = counter
#                 list_to_elem[i][counter] = tuple[i]
#                 counter += 1
#         list_size.append(counter)
#     return (list_to_int, list_to_elem, list_size)
#
# def convert_elem_of_states_to_int(list_of_lists_of_tuples):
#     """
#     returns states that are tuple of nonnegative integers, e.g.
#     (1,1,1,1,2) instead of ((0, 1.0), 'C', 1.0, 1.0, 0)
#     """
#
#     # flatten list of list of tuples to list of tuples
#     flatten_states = [item for list_of_tuples in
#                       list_of_lists_of_tuples for item in list_of_tuples]
#
#     global dict_elem_to_int, dict_int_to_elem, elem_set_sizes
#     dict_elem_to_int, dict_int_to_elem, elem_set_sizes = \
#         make_dict_of_elem(flatten_states)
#
#     new_all_states = []
#     for states in list_of_lists_of_tuples:
#         new_states = []
#         for state in states:
#             new_state = []
#             for i in range(len(state)):
#                 new_elem = dict_elem_to_int[i][state[i]]
#                 new_state.append(new_elem)
#             new_states.append(tuple(new_state))
#         new_all_states.append(new_states)
#     return new_all_states
