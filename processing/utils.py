from nltk.tree import Tree
import numpy as np
from copy import deepcopy, copy


def word_score_list_to_comp_score_list(orig_word_socres_list):
    # turn numpy array [word-score-1, word-score-2, ..]
    # into dictionary {1: word-score-1, 2: word-score-2, ...}
    comp_scores_list = {}
    for i in range(0, orig_word_socres_list.size):
        comp_scores_list[i] = orig_word_socres_list[i]
    return comp_scores_list


def build_leaf_nodes(word_socres_list, sent):
    nodes = []
    for i in range(len(word_socres_list)):
        nodes.append(Tree("{:.2f}".format(word_socres_list[i]), [sent[i]]))
    return nodes


def tree_to_str(tree):
    parse_string = ' '.join(str(tree).split())
    return parse_string


def fill_comp_list_w_previous(lists, num_words, filler_index=-1):
    # comps_list only record change (probably collapsed)

    num_iters = len(lists['comps_list'])
    new_comp_scores_list = [lists['comps_list'][0]]

    prev_list = None
    for i in range(1, num_iters):
        cur_list = copy(lists['comps_list'][i])
        if prev_list is None:
            new_comp_scores_list.append(cur_list)
        else:
            for j in range(0, num_words):
                if prev_list[j] != 0 and cur_list[j] == 0:
                    cur_list[j] = filler_index
            new_comp_scores_list.append(cur_list)

        prev_list = cur_list

    return new_comp_scores_list


def comp_to_tree(lists, sent):
    # return a parentheses based tree representation
    # NLTK will transform it into a binary tree. We go with left-branching for non-binary nodes
    # sent: a list of words
    # binarize=True: we use left binarization
    # main logic: we replicate the tree to fill the position of children!

    word_socres_list = word_score_list_to_comp_score_list(lists['scores_list'][0])

    children = build_leaf_nodes(word_socres_list, sent)

    num_iters = len(lists['comps_list'])
    num_words = len(sent)

    comps_list = fill_comp_list_w_previous(lists, num_words)

    prev_level = None
    for i in range(1, num_iters):
        comps = comps_list[i]  # lists['comps_list'][i]
        comp_scores = lists['comp_scores_list'][i]

        # print(comps)

        if prev_level is None:
            # we build at each level
            # [2 3 5], we ignore 0
            u_values = np.unique(comps[comps.nonzero()])
            # We build parent node, and then replace the children node
            # [0 1 2 3 ... 14] -> [0 t1 t1 3 ... 14]
            for v in u_values:
                # these two children are merged!
                child_indices = np.nonzero(comps == v)[0]
                parent_val = comp_scores[v]
                parent = Tree("{:.2f}".format(parent_val), [children[j] for j in child_indices])
                for c_i in child_indices:
                    children[c_i] = deepcopy(parent)
        else:
            # we compare difference between prev_level and current level
            # and only merge the difference
            carry_over_words = []
            for j in range(0, num_words):
                if comps[j] != 0 and prev_level[j] == 0:  # current build has a difference now
                    parent_val = comp_scores[comps[j]]
                    # now we check, for the prev_level, if this comps[j] is to left group or right group
                    # or both, if both, we construct a three nodes branch
                    if j == 0:  # first word
                        parent = Tree("{:.2f}".format(parent_val), [children[j], children[j + 1]])
                    elif j == num_words - 1:  # last word
                        parent = Tree("{:.2f}".format(parent_val), [children[j - 1], children[j]])
                    elif prev_level[j - 1] != 0 and prev_level[j + 1] != 0:
                        parent = Tree("{:.2f}".format(parent_val),
                                      [children[j - 1]] + carry_over_words + [children[j], children[j + 1]])
                        carry_over_words = []
                    elif prev_level[j - 1] != 0:
                        parent = Tree("{:.2f}".format(parent_val), [children[j - 1], children[j]])
                    elif prev_level[j + 1] != 0:
                        parent = Tree("{:.2f}".format(parent_val), carry_over_words + [children[j], children[j + 1]])
                        carry_over_words = []
                    else:
                        # this is the case where there are more than 1 item to join!
                        carry_over_words.append(children[j])
                        continue

                    child_indices = np.nonzero(comps == comps[j])[0]  # new group

                    for c_i in child_indices:
                        children[c_i] = parent

        prev_level = comps

    return children


# Note: This function is modified so intermediate nodes inherit parent's / original node's value
def chomsky_normal_form(tree, factor="right", horzMarkov=None, vertMarkov=0):
    # assume all subtrees have homogeneous children
    # assume all terminals have no siblings

    # A semi-hack to have elegant looking code below.  As a result,
    # any subtree with a branching factor greater than 999 will be incorrectly truncated.
    if horzMarkov is None: horzMarkov = 999

    # Traverse the tree depth-first keeping a list of ancestor nodes to the root.
    # I chose not to use the tree.treepositions() method since it requires
    # two traversals of the tree (one to get the positions, one to iterate
    # over them) and node access time is proportional to the height of the node.
    # This method is 7x faster which helps when parsing 40,000 sentences.

    nodeList = [(tree, [tree.label()])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node, Tree):

            # parent annotation
            parentString = ""
            originalNode = node.label()
            if vertMarkov != 0 and node != tree and isinstance(node[0], Tree):
                # parentString = "%s<%s>" % (parentChar, "-".join(parent))
                node.set_label(node.label())  # + parentString
                parent = [originalNode] + parent[:vertMarkov - 1]

            # add children to the agenda before we mess with them
            for child in node:
                nodeList.append((child, parent))

            # chomsky normal form factorization
            if len(node) > 2:
                childNodes = [child.label() for child in node]
                nodeCopy = node.copy()
                node[0:] = []  # delete the children

                curNode = node
                numChildren = len(nodeCopy)
                for i in range(1, numChildren - 1):
                    if factor == "right":
                        newHead = "%s" % originalNode  # "%s%s<%s>%s" % (originalNode, childChar, "-".join(childNodes[i:min([i+horzMarkov,numChildren])]),parentString) # create new head
                        newNode = Tree(newHead, [])
                        curNode[0:] = [nodeCopy.pop(0), newNode]
                    else:
                        newHead = "%s" % originalNode
                        # newHead = "%s%s<%s>%s" % (originalNode, childChar, "-".join(childNodes[max([numChildren-i-horzMarkov,0]):-i]),parentString)
                        newNode = Tree(newHead, [])
                        curNode[0:] = [newNode, nodeCopy.pop()]

                    curNode = newNode

                curNode[0:] = [child for child in nodeCopy]

if __name__ == '__main__':
    pass