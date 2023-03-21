import matplotlib.pyplot as plt
import numpy as np

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


def plot_prediction(test_labels, test_predictions):
    error = test_predictions.flatten() - test_labels.flatten()
    mean_error = np.mean(error)
    max_abs_error = max(np.abs(error))
    max_abs_test = max(np.abs(test_labels))
    max_value = max([max(test_labels), max(test_predictions)])
    min_value = max([max(test_labels), max(test_predictions)])
    
    fig, ax = plt.subplots()
    ax.scatter(test_labels, test_predictions, color="black", zorder=2)
    ax.plot([0, max_value], [0, max_value], 'r-', label='Predict equals to true value', zorder=1)
    plt.xlabel('True Value')
    plt.ylabel('Prediction')
    plt.xlim([0, 21/20 * max_value])
    plt.ylim([0, 21/20 * max_value])
    plt.legend()
    plt.show()
    
    fig, ax = plt.subplots()
    ax.scatter(test_labels, error, color="black", zorder=2)
    ax.plot([0, max_abs_test], [mean_error, mean_error], 'r-', label='Mean {0:.2f}'.format(mean_error), zorder=1)
    ax.plot([0, max_abs_test], [mean_error + 1.96*np.std(error), mean_error + 1.96*np.std(error)], 'b-', label='Mean + 1.96 σ {0:.2f}'.format(mean_error + 1.96*np.std(error)), zorder=1)
    ax.plot([0, max_abs_test], [mean_error - 1.96*np.std(error), mean_error - 1.96*np.std(error)], 'g-', label='Mean - 1.96 σ {0:.2f}'.format(mean_error - 1.96*np.std(error)), zorder=1)
    plt.xlabel('True Value')
    plt.ylabel('Prediction Error')
    plt.xlim([0, 21/20 * max_abs_test])
    plt.ylim([-21/20 * max_abs_error, 21/20 * max_abs_error])
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    counts, edges, bars = ax.hist(error, bins=20, range=[int(-21/20 * max_abs_error), int(21/20 * max_abs_error)], color="black")
    ax.set_xticks(edges.round(2))
    plt.xticks(rotation=70)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.show()