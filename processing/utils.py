from nltk import Tree

def chomsky_normal_form(tree, factor="right", horzMarkov=None, vertMarkov=0, childChar="|", parentChar="^"):
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
        if isinstance(node,Tree):

            # parent annotation
            parentString = ""
            originalNode = node.label()
            if vertMarkov != 0 and node != tree and isinstance(node[0],Tree):
                parentString = "%s<%s>" % (parentChar, "-".join(parent))
                node.set_label(node.label()) # + parentString
                parent = [originalNode] + parent[:vertMarkov - 1]

            # add children to the agenda before we mess with them
            for child in node:
                nodeList.append((child, parent))

            # chomsky normal form factorization
            if len(node) > 2:
                childNodes = [child.label() for child in node]
                nodeCopy = node.copy()
                node[0:] = [] # delete the children

                curNode = node
                numChildren = len(nodeCopy)
                for i in range(1,numChildren - 1):
                    if factor == "right":
                        newHead = "%s" % originalNode # "%s%s<%s>%s" % (originalNode, childChar, "-".join(childNodes[i:min([i+horzMarkov,numChildren])]),parentString) # create new head
                        newNode = Tree(newHead, [])
                        curNode[0:] = [nodeCopy.pop(0), newNode]
                    else:
                        newHead = "%s" % originalNode
                        # newHead = "%s%s<%s>%s" % (originalNode, childChar, "-".join(childNodes[max([numChildren-i-horzMarkov,0]):-i]),parentString)
                        newNode = Tree(newHead, [])
                        curNode[0:] = [newNode, nodeCopy.pop()]

                    curNode = newNode

                curNode[0:] = [child for child in nodeCopy]