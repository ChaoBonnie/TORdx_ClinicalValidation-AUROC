import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
np.random.seed(42)


def auroc_score(y_true, y_pred, conf_interval=0.95, n_bootstraps=1000):
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = (1.0 - conf_interval) / 2
    upper = lower + conf_interval
    confidence_lower = sorted_scores[int(lower * len(sorted_scores))]
    confidence_upper = sorted_scores[int(upper * len(sorted_scores))]
    return sorted_scores.mean(), (confidence_lower, confidence_upper)


outcomes = [22,40,66,25,64,48,25,22,50,69,77,43,40,22,61]
model_nums = [26,100,99,98,98,96,95,95,92,92,91,78,77,74,72,70,69,69,65,64,59,57,53,53,52,51,48,47,47,46,43,41,36,33,32,27,26,26,23,96,14,12,86]

# Considering outcome=2's
y_true = [0 if x == 0 else 1 for x in outcomes]
y_pred = [1 if x >= 26 else 0 for x in model_nums]
accuracy = len([x for x, y in zip(y_true, y_pred) if x == y]) / len(y_pred)
print('Accuracy: %.2f' % accuracy)

#Not considering outcome=2's
y_true = [0 if x == 0 else 1 for x in outcomes if x != 2]
y_pred = [1 if m >= 26 else 0 for m, x in zip(model_nums, outcomes) if x != 2]
accuracy = len([x for x, y in zip(y_true, y_pred) if x == y]) / len(y_pred)
print('Accuracy: %.2f' % accuracy)

# Not thresholding (and considering outcomes=2's)
y_true = [0 if x == 0 else 1 for x in outcomes]
y_pred = [m / 100 for m in model_nums]

# Not thresholding (and not considering outcomes=2's)
y_true = [0 if x == 0 else 1 for x in outcomes if x != 2]
y_pred = [m / 100 for m, x in zip(model_nums, outcomes) if x != 2]

y_true = np.array(y_true)
y_pred = np.array(y_pred)

conf_interval = 0.95
score, ci = auroc_score(y_true, y_pred, conf_interval=conf_interval)
print('Score: %.2f [%d%% CI: (%.2f, %.2f)]' % (score, int(100*conf_interval), ci[0], ci[1]))

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.set_title('ROC Curve')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
plt.show()