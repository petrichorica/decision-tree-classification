#global constants
ERROR = 0.00000001  # Multiplication and division operations might cause slight errors.


class Tree:
    
    def __init__(self, row=None, label=None, split_value=None, \
                smaller=None, larger=None, quality=None):
        '''
        row: list
        label: str
        split_value: float
        smaller, larger: Tree
        quality: True for "above 6", False for "not above 6"
        cost: int or float
        '''
        self.row = row
        self.label = label
        self.split_value = split_value
        self.smaller = smaller
        self.larger = larger
        self.quality = quality
        self.cost = None  # The overall cost R(t)
    
    def prune(self):
        self.label = None
        self.smaller = None
        self.larger = None
        self.split_value = None


def iterate_all_nodes(root):
        if root.smaller == None:  # root is a leaf node.
            return [root]
        else:
            return [root] + iterate_all_nodes(root.smaller) \
                + iterate_all_nodes(root.larger)


def get_data(file):

    def convert(x):  # Convert str into int or float
        x = x.strip()
        try:
            if '.' in x:
                x = float(x)
            else:
                x = int(x)
            return x
        except ValueError:
            return x

    with open(file, 'r') as f:
        data = []
        for line in f.readlines():
            x = line.split(",")
            y = []
            for item in x:
                y.append(convert(item))
            data.append(y)

        data = list(zip(*data))
        
    return data


def gini(prob_list):
    gini = prob_list[0] * prob_list[1]
    return gini


def get_impr(rows, dataset):
    p = get_probability(rows, dataset)
    i = gini([p, 1-p])
    return i


def get_probability(row_index_list, dataset):
    QUALITY = 6
    above = 0  # The number of wine data whose quality is above 6
    if len(row_index_list) == 0:
        return -1
    for i in row_index_list:
        if dataset[-1][i] > QUALITY:
            above += 1
    prob = above / len(row_index_list)
    return prob


def get_quality(rows, dataset):
    p = get_probability(rows, dataset)
    if p > 0.5:
        quality = True
    else:
        quality = False
    return quality


def right_classified(rows, dataset, quality):
    QUALITY = 6
    right = 0
    if quality:
        for row in rows:
            if dataset[-1][row] > QUALITY:
                right += 1
    else:
        for row in rows:
            if dataset[-1][row] <= QUALITY:
                right += 1
    return right


def classification(rows, dataset):
    split_label = None
    split_value = None
    smallest_impurity = 1

    for column in dataset[:-1]:
        label = column[0]
        values = []
        for row in rows:
            values.append(column[row])
        
        x = set(values)  # Check the number of different values
        if len(x) > 100:
            values = reduce_splitting_values(values)

        for value in values:
            row_s = []
            row_l = []
            for row in rows:
                if column[row] <= value:
                    row_s.append(row)
                else:
                    row_l.append(row)

            if len(row_l) > 0:
                p_s = len(row_s) / len(rows)
                p_l = len(row_l) / len(rows)
                impurity = p_s * get_impr(row_s, dataset)\
                    + p_l * get_impr(row_l, dataset)
                    
                if impurity < smallest_impurity:
                    smallest_impurity = impurity
                    split_label = label
                    split_value = value

    return split_label, split_value


def reduce_splitting_values(values):
    max_value = max(values)
    min_value = min(values)
    interval = (max_value - min_value) / 100
    values = []
    for i in range(101):
        value = min_value + i * interval
        values.append(value)
    return values


def grow_tree(rows, dataset):

    def divide(split_value, column):
        row_s = []
        row_l = []
        for row in rows:
            value = dataset[column][row]
            if value <= split_value:
                row_s.append(row)
            else:
                row_l.append(row)
        return row_s, row_l
    
    # Calculate impurity
    impurity = get_impr(rows, dataset)
    # Terminate condition 1: impurity = 0 (all the data belong to the same class).
    if impurity <= ERROR:
        return Tree(rows)

    # for testing
    if len(rows) <= 5:
        return Tree(rows)

    # for testing
    # p_t = len(rows) / len(ROWS)
    # I_t = p_t * impurity
    # if I_t <= 0.04:
    #     return Tree(rows)

    # Classify
    label, split_value = classification(rows, dataset)
    for index in range(len(dataset)):
        if dataset[index][0] == label:
            label_col = index
            break
    row_s, row_l = divide(split_value, label_col)
    # row_s and row_l cannot be 0.

    # Calculate the decrease of impurity
    p_s = len(row_s) / len(rows)
    p_l = 1 - p_s
    i_s = get_impr(row_s, dataset)
    i_l = get_impr(row_l, dataset)
    i_decrease = impurity - p_s*i_s - p_l*i_l

    # Terminate condition 2: the impurity does not decrease.
    if i_decrease == 0:
        return Tree(rows)
    # For debugging
    elif i_decrease < 0:
        print('Something is wrong!')

    smaller = grow_tree(row_s, dataset)
    larger = grow_tree(row_l, dataset)

    return Tree(rows, label, split_value, smaller, larger)
    

def save(tree, filename):

    if isinstance(tree, Tree):
        q = [tree]
        while q:
            t = q[0]
            del q[0]
            if t.smaller != None:
                q.append(t.smaller)
            if t.larger != None:
                q.append(t.larger)
            with open(filename, 'a') as f:
                if t.label != None:
                    f.write(t.label+' ')
                if t.split_value != None:
                    f.write(str(t.split_value)+'\n')
    elif isinstance(tree, None):
        return
    else:
        print('Wrong input')
        return


def get_leaf_nodes(tree_node):
    leaf = []
    if isinstance(tree_node, Tree):
        if tree_node.smaller == None:
            leaf.append(tree_node)
        else:
            leaf += get_leaf_nodes(tree_node.smaller)
            leaf += get_leaf_nodes(tree_node.larger)
    else:
        print('wrong input')
    return leaf


def define_quality(tree, dataset):

    if isinstance(tree, Tree):
        q = [tree]
        while q:
            t = q[0]
            del q[0]
            t.quality = get_quality(t.row, dataset)
            # If t.smaller or t.right is None, t is a leaf node.
            if t.smaller == None:
                continue
            else:
                q.append(t.smaller)
                q.append(t.larger)
    else:
        print('Wrong input')
    return tree


# Classification cost R(t)
def cost_of_leaf_node(tree, leaf_node, dataset):
    if isinstance(leaf_node, Tree) and isinstance(tree, Tree):
        r = get_probability(leaf_node.row, dataset)
        if r > 0.5:
            r = 1 - r
        p = len(leaf_node.row) / len(tree.row)
        R = p * r
        return R
    else:
        print('Wrong input')
        return


def calculate_cost_of_each_node(tree, dataset):
    if isinstance(tree, Tree):
        q = [tree]
        while q:
            t = q[0]
            del q[0]
            # leaf = get_leaf_nodes(t)
            R = cost_of_leaf_node(tree, t, dataset)
            t.cost = R
            # print(t.label, t.split_value, t.cost)

            if t.smaller != None:
                q.append(t.smaller)
                q.append(t.larger)
    else:
        print('Wrong input')
    return tree


def cost_of_pruned_subtree(tree, dataset, terminal_node=None):
    if isinstance(tree, Tree):
        if terminal_node:
            leaf_T = get_leaf_nodes(tree)
            leaf_t = []
            for t in terminal_node:
                leaf_t += get_leaf_nodes(t)
            leaf = []
            for l in leaf_T:
                if l not in leaf_t:
                    leaf.append(l)
            leaf += terminal_node
        else:
            leaf = get_leaf_nodes(tree)

        R_T = 0
        for l in leaf:
            R_t = l.cost
            R_T += R_t
        return R_T
    else:
        print('Wrong input')
        return


def pruning_step_1(tree, dataset):
    if isinstance(tree, Tree):
        q = [tree]
        while q:
            t = q[0]
            del q[0]
            if t.smaller != None:
                R_t = t.cost
                R_t_s = t.smaller.cost
                R_t_l = t.larger.cost
                if R_t + ERROR <= (R_t_s + R_t_l):
                    t.prune()
                else:
                    q.append(t.smaller)
                    q.append(t.larger)
        return tree
    else:
        print('Wrong input')
        return


def alpha(tree_node, dataset, terminal_node=[]):
    # tree_node must not be leaf node.
    if isinstance(tree_node, Tree):
        R_t = tree_node.cost
        # The cost of the subtree whose root is tree_node
        R_t_root = cost_of_pruned_subtree(tree_node, dataset, terminal_node)
        num_of_leaf = num_of_leaf_nodes(tree_node, terminal_node)
        alpha = (R_t - R_t_root) / (num_of_leaf - 1)
        return alpha
    else:
        print('Wrong input')
        return


def num_of_leaf_nodes(root, terminal_node=[]):
    if terminal_node:
        leaf = []
        q = [root]
        while q:
            t = q[0]
            del q[0]
            # If t is not a leaf node
            if (t.smaller != None) and (t not in terminal_node):
                q.append(t.smaller)
                q.append(t.larger)
            else:
                leaf.append(t)
        return len(leaf)
    else:
        leaf = get_leaf_nodes(root)
        return len(leaf)


def get_pruning_point(tree, dataset, terminal_node=[]):
    # terminal_node: list
    # The function does not calculate alpha of the child_nodes of the terminal_node.
    if isinstance(tree, Tree):
        min_alpha = 100
        prun_t = []
        all_nodes = iterate_all_nodes(tree)
        leaf_nodes = get_leaf_nodes(tree)
        nonleaf = []
        for t in all_nodes:
            if t in leaf_nodes:
                continue
            elif t in terminal_node:
                continue
            else:
                nonleaf.append(t)

        for t in nonleaf:
            a = alpha(t, dataset, terminal_node)
            if a >= 0:
                if a + ERROR < min_alpha:
                    prun_t = [t]
                    min_alpha = a
                elif -ERROR < (min_alpha-a) < ERROR:  # a = min_alpha
                    prun_t.append(t)
                
        return min_alpha, prun_t
        # If tree is a leaf node, min_alpha=100, prun_t=[].
    else:
        print('Wrong input')
        return


def get_alpha_list(tree, train_data):
    alpha_list = [0]
    pruning_nodes = []
    if isinstance(tree, Tree):
        while True:
            a, t_list = get_pruning_point(tree, train_data, pruning_nodes)
            if not t_list:  # t_list is empty.
                break
            alpha_list.append(a)
            pruning_nodes += t_list
        return alpha_list


def get_alpha_prime_list(alpha_list):
    alpha_prime = []
    for i in range(len(alpha_list)-1):
        a_prime = (alpha_list[i] * alpha_list[i+1]) ** 0.5
        alpha_prime.append(a_prime)
    return alpha_prime


def divide_data(row_index_list, k):
    divided_len = len(row_index_list) // k
    divide_point = 0
    data_fractions = []
    for i in range(k-1):
        data_fractions.append(row_index_list[divide_point:(divide_point+divided_len)])
        divide_point += divided_len
    data_fractions.append(row_index_list[divide_point:])
    divided_data = []
    for i in range(k):
        data_list = []
        for j in range(k):
            if i != j:
                data_list += (data_fractions[j])
        divided_data.append(data_list)
    return divided_data, data_fractions


def cost_if_prune_at_alpha(tree, p_alpha, train_data, test_data):
    if isinstance(tree, Tree):
        all_nodes = iterate_all_nodes(tree)
        leaf_nodes = get_leaf_nodes(tree)
        nonleaf = []
        for t in all_nodes:
            if t in leaf_nodes:
                continue
            else:
                nonleaf.append(t)
        pruned_nodes = []
        for t in nonleaf:
            a = alpha(t, train_data)
            if 0 <= a <= p_alpha:
                pruned_nodes.append(t)
        right = right_classified_in_a_tree(test_data, tree, pruned_nodes)
        misclassified = len(test_data[0]) - 1 - right
        return misclassified


def prune_at_alpha(tree, p_alpha, train_data):
    if isinstance(tree, Tree):
        all_nodes = iterate_all_nodes(tree)
        leaf_nodes = get_leaf_nodes(tree)
        nonleaf = []
        for t in all_nodes:
            if t in leaf_nodes:
                continue
            else:
                nonleaf.append(t)
        pruned_nodes = []
        for t in nonleaf:
            a = alpha(t, train_data)
            if a <= p_alpha:
                pruned_nodes.append(t)
        for t in pruned_nodes:
            t.prune()


# Cross validation
def prune(tree, train_data_set, k=3):
    train_data_set_row = list(zip(*train_data_set))
    if isinstance(tree, Tree):
        alpha_list = get_alpha_list(tree, train_data_set)
        alpha_prime_list = get_alpha_prime_list(alpha_list)
        v_tree = []
        v_row, row_test = divide_data(tree.row, k)
        for i in range(k):
            rows = v_row[i]
            t = grow_tree(rows, train_data_set)
            t = define_quality(t, train_data_set)
            t = calculate_cost_of_each_node(t, train_data_set)
            v_tree.append(t)
        
        min_cost = 1000
        best_alpha = 0
        for a in alpha_prime_list:
            cost_sum = 0
            for i in range(k):
                test_data_rows = row_test[i]
                test_data_set = [train_data_set_row[0]]
                for j in test_data_rows:
                    test_data_set.append(train_data_set_row[j])
                test_data_set = list(zip(*test_data_set))
                tree_test = v_tree[i]
                cost = cost_if_prune_at_alpha(tree_test, a, train_data_set, test_data_set)
                cost_sum += cost
                # cost += accr(tree, test_data_set)
                # print(cost)
            if cost_sum < min_cost:
                best_alpha = a
                min_cost = cost_sum
            
        
        prune_at_alpha(tree, best_alpha, train_data_set)


def accr(tree, test_data):
    if isinstance(tree, Tree):
        right = right_classified_in_a_tree(test_data, tree)
        accr = right / (len(test_data[0])-1)
        return accr
    else:
        print('Wrong input')
        return


def right_classified_in_a_tree(test_data, tree, terminal_node=None):
    test_data_row = list(zip(*test_data))
    right = 0
    if isinstance(tree, Tree):
        if not terminal_node:  # terminal_node is empty.
            for row in test_data_row[1:]:
                t = tree
                # When t is not a leaf node:
                while t.smaller != None:
                    for col in range(len(test_data)):
                        if test_data[col][0] == t.label:
                            split_col = col
                            break
                    if row[split_col] <= t.split_value:
                        t = t.smaller
                    else:
                        t = t.larger
                # print(row[-1])
                if (row[-1] > 6) == t.quality:
                    right += 1
        else:
            for row in test_data_row[1:]:
                t = tree
                # When t is not a leaf node:
                while (t.smaller != None) and (t not in terminal_node):
                    for col in range(len(test_data)):
                        if test_data[col][0] == t.label:
                            split_col = col
                            break
                    if row[split_col] <= t.split_value:
                        t = t.smaller
                    else:
                        t = t.larger
                # print(row[-1])
                if (row[-1] > 6) == t.quality:
                    right += 1
    return right


# Data processing
d = get_data("train.csv")
d_test = get_data("test.csv")

row_len = len(d[0])
rows = list(range(1, row_len))
ROWS = rows

# Grow a tree
t = grow_tree(rows, d)
t = define_quality(t, d)
t = calculate_cost_of_each_node(t, d)

print('Accuracy before pruning:', end=' ')
print(accr(t, d_test))

# Save the tree
filename = 'tree_without_pruning.txt'
with open(filename, 'w') as f:
    f.write('Tree with pruning\n')
save(t, filename)

# Prune the tree
t = pruning_step_1(t, d)
prune(t, d, 5)

# save the tree
filename = 'tree_with_pruning.txt'
with open(filename, 'w') as f:
    f.write('Tree with pruning\n')
save(t, filename)

print('Accuracy after pruning:', end=' ')
print(accr(t, d_test))
