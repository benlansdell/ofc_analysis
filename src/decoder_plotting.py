import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
    ax.set(xlabel='Predicted', ylabel='True', xticklabels=classes, yticklabels=classes)
    if title:
        ax.set_title(title)
    return fig, ax

def plot_accuracy_by_group(test_accuracy_by_group, group_to_maze_choice, fn_in):
    #Plot test_accuracy_by_group
    keys, vals = zip(*test_accuracy_by_group.items())
    trial_names = [k.split('Rds-')[1] for k in keys]
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    ax.bar(range(len(test_accuracy_by_group)), vals)
    #Change x labels to be the keys
    ax.set_xticks(range(len(test_accuracy_by_group)), trial_names, rotation=45);

    #Plot the maze choice for each group as a vspan for when maze_choice is 1
    for i, (k, v) in enumerate(group_to_maze_choice.items()):
        c = 'blue' if v == 'right' else 'red'
        ax.axvspan(i-0.5, i+0.5, facecolor=c, alpha=0.2)

    ax.set_xlabel('Day/trial')
    ax.set_ylabel('Accuracy')
    ax.set_title("Accuracy by day/trial and maze choice (red = left)")
    fig.suptitle(fn_in.split('/')[-1].split('.')[0])
    