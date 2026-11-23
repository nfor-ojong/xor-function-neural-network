\# Non-Linear XOR Classification with PyTorch

&nbsp;

An end-to-end binary classification project implementing a multi-layer perceptron (MLP) in PyTorch to separate a two-dimensional XOR dataset. Because opposing quadrants in an XOR distribution are not linearly separable, a single linear boundary cannot classify the data. This repository walks through data exploration, leak-free feature scaling, neural network training, and held-out test evaluation with full visual diagnostics.

&nbsp;

\#\# Dataset & Problem Formulation

&nbsp;

The dataset contains two continuous input features (\`x1\`, \`x2\`) and a binary target (\`class\`) split across \`data/train.csv\` and \`data/test.csv\` (80 test samples):

&nbsp;

\- \*\*Signal (\`1\`):\*\* Samples falling in Quadrants I and III (both coordinates share the same sign).

\- \*\*Background (\`0\`):\*\* Samples falling in Quadrants II and IV (coordinates have opposite signs).

&nbsp;

\!\[Training Data Scatter Plot\](plots/train\_data\_scatter\_plot.png)

&nbsp;

\#\# Project Layout

&nbsp;

\`\`\`text

xor-function-neural-network/

├── data/

│   ├── train.csv                              \# Training and validation dataset

│   └── test.csv                               \# Held-out test dataset (80 rows)

├── model/

│   ├── neural\_network.pth                     \# Saved PyTorch state\_dict weights

│   └── standard\_scaler.joblib                 \# Fitted StandardScaler parameters

├── notebooks/

│   ├── 01\_model\_training\_and\_tuning.ipynb     \# Data prep, architecture, training loop, and export

│   └── 02\_model\_evaluation\_and\_metrics.ipynb  \# Inference, test metrics, ROC curve, and decision boundary

├── plots/

│   ├── train\_data\_scatter\_plot.png            \# 2D scatter plot of training samples

│   ├── train\_validation\_loss\_evolution.png    \# Training vs. validation BCE loss curve

│   ├── confusion\_matrix\_on\_test.png           \# Test set confusion matrix

│   ├── roc\_curve\_on\_test.png                  \# Test set ROC curve

│   └── decision\_boundary\_on\_test.png          \# Learned non-linear decision surface

├── test-results/

│   └── test\_metrics.png                       \# Console summary of test evaluation metrics

├── requirements.txt                           \# Pinned project dependencies

└── README.md

\`\`\`

&nbsp;

\#\# Model Design & Training Workflow

&nbsp;

All preprocessing and training logic is contained in \`notebooks/01\_model\_training\_and\_tuning.ipynb\`:

&nbsp;

\- \*\*Leak-Free Scaling:\*\* \`StandardScaler\` is fitted strictly on the training split and saved to \`model/standard\_scaler.joblib\` so identical transformation statistics are applied during test inference.

\- \*\*Network Architecture:\*\* Built with \`nn.Sequential\`:

&nbsp;&nbsp;\- \`Linear(2 \-\> 4)\` \+ \`ReLU\`

&nbsp;&nbsp;\- \`Linear(4 \-\> 4)\` \+ \`ReLU\`

&nbsp;&nbsp;\- \`Linear(4 \-\> 1)\` (outputs raw logits)

\- \*\*Loss & Reproducibility:\*\* Trained using \`nn.BCEWithLogitsLoss()\` with \`torch.manual\_seed(42)\`. Learned parameters are exported via \`model.state\_dict()\` to \`model/neural\_network.pth\`.

&nbsp;

\!\[Training and Validation Loss Evolution\](plots/train\_validation\_loss\_evolution.png)

&nbsp;

\#\# Evaluation & Test Results

&nbsp;

Final evaluation is isolated in \`notebooks/02\_model\_evaluation\_and\_metrics.ipynb\`. The notebook loads \`data/test.csv\`, applies \`standard\_scaler.joblib\`, restores \`neural\_network.pth\`, converts output logits to probabilities with \`torch.sigmoid\`, and classifies samples at a \`0.5\` decision threshold.

&nbsp;

\!\[Test Performance Metrics\](test-results/test\_metrics.png)

&nbsp;

\| Metric \| Score \|

\| :--- \| :---: \|

\| \*\*Accuracy\*\* \| 100.0% \|

\| \*\*Precision\*\* \| 100.0% \|

\| \*\*Recall\*\* \| 100.0% \|

\| \*\*F1-Score\*\* \| 100.0% \|

&nbsp;

\#\#\# Confusion Matrix & ROC Curve

&nbsp;

The network classifies all 80 unseen test samples without false positives or false negatives.

&nbsp;

\| Confusion Matrix \| ROC Curve \|

\| :---: \| :---: \|

\| \!\[Confusion Matrix on Test\](plots/confusion\_matrix\_on\_test.png) \| \!\[ROC Curve on Test\](plots/roc\_curve\_on\_test.png) \|

&nbsp;

\#\#\# Decision Boundary

&nbsp;

Evaluating the trained model over a 2D meshgrid across the input domain shows how the two ReLU hidden layers partition the four XOR quadrants cleanly:

&nbsp;

\!\[Decision Boundary on Test\](plots/decision\_boundary\_on\_test.png)

&nbsp;

\#\# Running Locally

&nbsp;

1\. \*\*Clone the repository:\*\*

&nbsp;&nbsp;&nbsp;\`\`\`bash

&nbsp;&nbsp;&nbsp;git clone https\://github.com/nfor-ojong/xor-function-neural-network.git

&nbsp;&nbsp;&nbsp;cd xor-function-neural-network

&nbsp;&nbsp;&nbsp;\`\`\`

&nbsp;

2\. \*\*Install dependencies:\*\*

&nbsp;&nbsp;&nbsp;\`\`\`bash

&nbsp;&nbsp;&nbsp;pip install \-r requirements.txt

&nbsp;&nbsp;&nbsp;\`\`\`

&nbsp;

3\. \*\*Open the notebooks:\*\*

&nbsp;&nbsp;&nbsp;Launch Jupyter or VS Code and run \`notebooks/01\_model\_training\_and\_tuning.ipynb\` followed by \`notebooks/02\_model\_evaluation\_and\_metrics.ipynb\`.