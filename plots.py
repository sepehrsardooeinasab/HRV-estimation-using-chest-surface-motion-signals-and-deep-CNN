import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_history(history):
    fig, ax = plt.subplots()    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.plot(history.epoch, np.array(history.history['loss']), 
           label='Train', color="gray")
    ax.plot(history.epoch, np.array(history.history['val_loss']),
           label = 'Val', color="black")
    plt.legend()
    plt.show()


def plot_prediction(test_labels, test_predictions, margin = 1.1):
    error = test_predictions.flatten() - test_labels.flatten()
    mean_error = np.mean(error)
    max_abs_error = max(np.abs(error))
    max_abs_test = max(np.abs(test_labels))
    max_value = max([max(test_labels), max(test_predictions)])
    min_value = min([min(test_labels), min(test_predictions)])
    margin_min = 1/1.1 if min_value>0 else margin
    
    fig, ax = plt.subplots()
    ax.scatter(test_labels, test_predictions, color="black", zorder=2, s=50)
    ax.plot([min_value * margin_min, max_value * margin], [min_value * margin_min, max_value * margin], 'r-', label='predicted value equal the true value', zorder=1, linewidth=5)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel('Reference Values')
    plt.ylabel('Predicted Values')
    plt.xlim([min_value * margin_min, margin * max_value])
    plt.ylim([min_value * margin_min, margin * max_value])
    #plt.legend()
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(np.mean(np.array([test_labels.flatten(), test_predictions.flatten()]), axis=0 ), error, color="black", zorder=2, s=50)
    ax.plot([min_value * margin_min, max_abs_test * margin], [mean_error, mean_error], 'r-', label='Mean = {0:.2f}'.format(mean_error), zorder=1, linewidth=5)
    ax.plot([min_value * margin_min, max_abs_test * margin], [mean_error + 1.96*np.std(error), mean_error + 1.96*np.std(error)], 'b-', label='Mean + 1.96 σ = {0:.2f}'.format(mean_error + 1.96*np.std(error)), zorder=1, linewidth=5)
    ax.plot([min_value * margin_min, max_abs_test * margin], [mean_error - 1.96*np.std(error), mean_error - 1.96*np.std(error)], 'g-', label='Mean - 1.96 σ = {0:.2f}'.format(mean_error - 1.96*np.std(error)), zorder=1, linewidth=5)
    plt.xlabel('Mean of Predicted and Reference Values')
    plt.ylabel('Reference - Predicted Values')
    plt.xlim([min_value * margin_min, margin * max_abs_test])
    plt.ylim([-margin * max_abs_error, margin * max_abs_error])
    #plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    counts, edges, bars = ax.hist(error, bins=20, range=[int(-margin * max_abs_error), int(margin * max_abs_error)], color="black")
    ax.set_xticks(edges.round(2))
    plt.xticks(rotation=70)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.show()