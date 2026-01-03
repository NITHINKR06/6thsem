export interface Topic {
  id: string
  title: string
  content: {
    explanation: string[]
    keyPoints?: string[]
    diagrams?: {
      title: string
      description: string
      imageUrl?: string
    }[]
    examples?: {
      title: string
      problem: string
      solution: string
    }[]
    problems?: {
      question: string
      answer: string
    }[]
    examTips?: string[]
    resources?: {
      title: string
      url: string
      description: string
    }[]
  }
}

export interface Unit {
  id: string
  title: string
  description: string
  topics: Topic[]
}

export interface Subject {
  id: string
  title: string
  shortTitle: string
  description: string
  units: Unit[]
}

export const subjectsData: Record<string, Subject> = {
  "ml-cyber": {
    id: "ml-cyber",
    title: "Machine Learning for Cyber Security",
    shortTitle: "ML for Cyber",
    description:
      "Apply machine learning algorithms to cybersecurity challenges including threat detection, malware classification, and anomaly detection.",
    units: [
      {
        id: "unit-1",
        title: "Introduction to Machine Learning, Decision Trees & Neural Networks",
        description: "Core ML concepts, supervised/unsupervised learning, decision tree algorithms, and neural network fundamentals for cyber security applications",
        topics: [
          {
            id: "intro-to-ml",
            title: "Introduction to Machine Learning",
            content: {
              explanation: [
                "Machine Learning (ML) is a branch of Artificial Intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and improve their performance over time.",
                "Why Machine Learning? Traditional programming requires explicit rules for every scenario, which becomes impractical for complex problems like spam detection, fraud identification, or malware classification. ML overcomes this by automatically discovering patterns in data, adapting to new threats, and handling scenarios that weren't explicitly programmed.",
                "Types of Machine Learning: (1) Supervised Learning - learns from labeled training data to predict outcomes for new data. Examples: classification (spam/not spam), regression (predicting values). (2) Unsupervised Learning - discovers hidden patterns in unlabeled data. Examples: clustering (grouping similar attacks), anomaly detection. (3) Reinforcement Learning - learns through trial and error with rewards/penalties.",
                "Supervised Learning uses labeled datasets where each example has input features and a known output label. The algorithm learns the mapping from inputs to outputs. Used in intrusion detection (normal/attack), malware classification (benign/malicious), and phishing detection.",
                "Unsupervised Learning works with unlabeled data to find hidden structures. Clustering groups similar data points together (e.g., grouping malware families). Anomaly detection identifies unusual patterns that may indicate security threats. No predefined labels are needed.",
              ],
              keyPoints: [
                "ML enables computers to learn patterns from data without explicit programming",
                "Supervised Learning: Uses labeled data, predicts known categories (classification/regression)",
                "Unsupervised Learning: Uses unlabeled data, discovers hidden patterns (clustering/anomaly detection)",
                "Key difference: Supervised needs labeled training data; Unsupervised finds patterns in unlabeled data",
                "In Cyber Security: Supervised for known threat detection, Unsupervised for zero-day attacks",
                "Training Data → Model → Predictions is the basic ML workflow",
              ],
              diagrams: [
                {
                  title: "Supervised Learning Example: Linear Regression",
                  description: "Visual representation of Linear Regression, a fundamental supervised learning algorithm used for predicting continuous values.",
                  imageUrl: "/diagrams/linear_regression.svg",
                },
                {
                  title: "Unsupervised Learning Example: K-Means Clustering",
                  description: "Visual representation of K-Means Clustering, a popular unsupervised learning algorithm used for grouping similar data points.",
                  imageUrl: "/diagrams/kmeans_clustering.svg",
                },
              ],
              examples: [
                {
                  title: "Spam Email Classification (Supervised Learning)",
                  problem: "Given 10,000 emails labeled as 'spam' or 'not spam', train a model to classify new emails.",
                  solution: "This is a supervised learning classification problem.\n1. Training Data: 10,000 labeled emails\n2. Features: Word frequency, sender reputation, link count, header patterns\n3. Algorithm: Naive Bayes or Random Forest\n4. Output: Binary classification (spam = 1, not spam = 0)\n5. The model learns patterns that distinguish spam from legitimate emails.",
                },
                {
                  title: "Network Anomaly Detection (Unsupervised Learning)",
                  problem: "Detect unusual network behavior without prior labels of what constitutes an attack.",
                  solution: "This is an unsupervised learning problem.\n1. Data: Network traffic logs (no labels)\n2. Approach: Clustering or Isolation Forest\n3. Process: Learn 'normal' traffic patterns\n4. Detection: Flag traffic that deviates significantly from normal\n5. Advantage: Can detect zero-day attacks not seen before.",
                },
              ],
              problems: [
                {
                  question: "Define Machine Learning and explain why it is important for cyber security. (2 marks)",
                  answer: "Machine Learning is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed. In cyber security, ML is important because: (1) It can detect previously unknown (zero-day) attacks by learning patterns, (2) It reduces manual effort in analyzing large volumes of security data, (3) It adapts to evolving threats automatically, (4) It reduces false positives compared to rule-based systems.",
                },
                {
                  question: "Differentiate between Supervised and Unsupervised Learning with examples. (5 marks)",
                  answer: "Supervised Learning: Uses labeled training data where output is known. Algorithm learns mapping from input to output. Examples: Email spam detection (spam/not spam labels), Malware classification (malicious/benign), Intrusion detection with known attack types.\n\nUnsupervised Learning: Uses unlabeled data where output is not known. Algorithm discovers hidden patterns or groupings. Examples: Clustering malware into families, Detecting anomalies in network traffic, Grouping similar user behaviors.\n\nKey Differences: (1) Supervised needs labels, Unsupervised doesn't, (2) Supervised predicts, Unsupervised discovers, (3) Supervised for known threats, Unsupervised for unknown patterns.",
                },
              ],
              examTips: [
                "Always define ML before explaining types - 'learning from data without explicit programming'",
                "Memorize at least 2 cyber security examples for each ML type",
                "Supervised = Teacher present (labels), Unsupervised = No teacher",
                "Common exam question: 'Differentiate supervised and unsupervised learning' - use table format",
                "Keywords to include: labeled/unlabeled data, classification, clustering, training, prediction",
              ],
            },
          },
          {
            id: "decision-trees",
            title: "Decision Tree Learning",
            content: {
              explanation: [
                "A Decision Tree is a supervised machine learning algorithm that uses a tree-like hierarchical structure to make classification or regression decisions. It works like a flowchart where each internal node represents a test on an attribute, each branch represents the outcome of that test, and each leaf node represents a class label (final decision).",
                "Decision Tree Representation: The tree consists of four components: (1) Root Node - the topmost node representing the first attribute test (attribute with highest information gain), (2) Internal Nodes - decision points that test an attribute, (3) Branches/Edges - represent possible values of an attribute connecting nodes, (4) Leaf Nodes - terminal nodes containing the final class labels.",
                "Appropriate Problems for Decision Tree Learning: Decision trees work best when: (1) Instances are represented by attribute-value pairs, (2) Target function has discrete output values (classification), (3) Disjunctive descriptions may be required (conditions combined with OR), (4) Training data may contain noise/errors, (5) Training data may have missing attribute values. Example: Classifying network traffic as 'Attack' or 'Normal' based on attributes like protocol, port, packet size.",
                "Basic Decision Tree Learning Algorithm (ID3): ID3 (Iterative Dichotomiser 3) is a greedy, top-down algorithm that builds decision trees. Core idea: At each node, select the attribute that best classifies the training examples. 'Best' is determined using Information Gain.",
                "Entropy measures the impurity or uncertainty in a dataset. Formula: H(S) = -Σ pᵢ log₂(pᵢ) where pᵢ is the proportion of class i. Entropy = 0 means pure set (all same class), Entropy = 1 means maximum impurity (for binary: 50-50 split).",
                "Information Gain measures the reduction in entropy after splitting on an attribute. Formula: Gain(S,A) = H(S) - Σ(|Sᵥ|/|S|) × H(Sᵥ) where A is the attribute, and Sᵥ is subset where A has value v. The attribute with HIGHEST information gain is selected as the decision node.",
                "ID3 Algorithm Steps: (1) Calculate entropy of entire dataset, (2) For each attribute, calculate information gain, (3) Select attribute with highest gain as decision node, (4) Create branches for each value of selected attribute, (5) Recursively repeat for each subset until: all examples same class (create leaf), no attributes left (majority class leaf), or no examples (parent's majority class).",
              ],
              keyPoints: [
                "Decision Tree: Supervised learning, tree-structure, if-then rules",
                "Components: Root Node, Internal Nodes, Branches, Leaf Nodes",
                "ID3 Algorithm: Top-down, greedy search, uses entropy and information gain",
                "Entropy Formula: H(S) = -Σ pᵢ log₂(pᵢ) — measures impurity (0=pure, 1=max impure)",
                "Information Gain: Gain(S,A) = H(S) - Weighted average of subset entropies",
                "Best Attribute = Highest Information Gain",
                "Appropriate when: discrete outputs, attribute-value pairs, may have noise/missing values",
                "Greedy approach: locally optimal choice at each node, no backtracking",
              ],
              diagrams: [
                {
                  title: "Decision Tree Structure",
                  description: "Draw a tree with: Root Node at top labeled 'Attribute A?' with two branches (Yes/No). Each branch leads to either an Internal Node 'Attribute B?' or a Leaf Node with class label. Label all four components clearly: Root Node, Internal Node, Branch/Edge, Leaf Node. Show that root tests first attribute, branches show attribute values, leaves contain final classification.",
                  imageUrl: "/diagrams/decision_tree_external.svg",
                },
                {
                  title: "ID3 Algorithm Flowchart",
                  description: "Draw a flowchart: START → Check 'All examples same class?' (if YES → Return leaf with that class) → Check 'Attributes empty?' (if YES → Return majority class leaf) → Calculate Entropy H(S) → Calculate Information Gain for each attribute → Select attribute with MAX gain → Create decision node → For each value: create branch, create subset, RECURSE → END. Show the recursive nature with an arrow back to the entropy calculation step.",
                  imageUrl: "/diagrams/id3_flowchart.svg",
                },
                {
                  title: "Entropy and Information Gain Calculation",
                  description: "Draw a dataset box showing mixed classes (e.g., 9 Normal, 5 Attack). Show Entropy calculation with formula. Then show splitting by an attribute into two subsets. Calculate entropy of each subset. Show weighted average calculation. Finally show Information Gain = Original Entropy - Weighted Average. Highlight that higher gain = better attribute.",
                  imageUrl: "/diagrams/entropy_calc.svg",
                },
              ],
              examples: [
                {
                  title: "Decision Tree for Intrusion Detection",
                  problem: "Build a decision tree for the following network traffic data:\n\n| ID | Protocol | Port | Class |\n|----|----------|------|-------|\n| 1 | TCP | Known | Normal |\n| 2 | UDP | Unknown | Attack |\n| 3 | TCP | Known | Normal |\n| 4 | UDP | Known | Attack |\n| 5 | TCP | Unknown | Normal |\n| 6 | UDP | Unknown | Attack |",
                  solution: "Step 1: Calculate Entropy(S)\nTotal=6, Normal=3, Attack=3\nH(S) = -(3/6)log₂(3/6) - (3/6)log₂(3/6) = -0.5(-1) - 0.5(-1) = 1.0\n\nStep 2: Information Gain for Protocol\nTCP: {1,3,5} → 3 Normal, 0 Attack → H=0 (pure)\nUDP: {2,4,6} → 0 Normal, 3 Attack → H=0 (pure)\nGain(Protocol) = 1.0 - [(3/6)×0 + (3/6)×0] = 1.0\n\nStep 3: Protocol has Gain=1.0 (perfect split!)\n\nFinal Tree:\n        [Protocol?]\n        /        \\\n      TCP        UDP\n       ↓          ↓\n   [NORMAL]   [ATTACK]",
                },
                {
                  title: "Calculating Entropy and Information Gain",
                  problem: "Dataset has 14 samples: 9 'Yes' and 5 'No'. Calculate the entropy.",
                  solution: "Given: Total = 14, Yes = 9, No = 5\n\np(Yes) = 9/14 = 0.643\np(No) = 5/14 = 0.357\n\nEntropy H(S) = -(9/14)log₂(9/14) - (5/14)log₂(5/14)\n            = -(0.643)(-0.637) - (0.357)(-1.486)\n            = 0.410 + 0.530\n            = 0.940\n\nInterpretation: Entropy ≈ 0.94 indicates high impurity (close to 1), meaning the dataset is quite mixed between Yes and No classes.",
                },
              ],
              problems: [
                {
                  question: "Define Decision Tree and list its four main components. (3 marks)",
                  answer: "Definition: A Decision Tree is a supervised machine learning algorithm that uses a tree-like hierarchical structure to make classification decisions based on attribute tests.\n\nFour Components:\n1. Root Node - Topmost node; first attribute test (highest info gain)\n2. Internal Nodes - Decision points that test an attribute\n3. Branches/Edges - Connect nodes; represent attribute values\n4. Leaf Nodes - Terminal nodes containing final class labels",
                },
                {
                  question: "Explain the ID3 algorithm for constructing a decision tree. (5 marks)",
                  answer: "ID3 (Iterative Dichotomiser 3) Algorithm:\n\n1. Calculate Entropy of dataset: H(S) = -Σ pᵢ log₂(pᵢ)\n\n2. For each attribute A, calculate Information Gain:\n   Gain(S,A) = H(S) - Σ(|Sᵥ|/|S|) × H(Sᵥ)\n\n3. Select attribute with HIGHEST Information Gain\n\n4. Create decision node for selected attribute\n\n5. Create branches for each value of the attribute\n\n6. Recursively apply steps 1-5 on each subset\n\n7. Stop when:\n   - All examples same class → create leaf\n   - No attributes remaining → majority class leaf\n   - No examples → parent's majority class\n\nKey: ID3 uses greedy, top-down search - makes locally optimal choice without backtracking.",
                },
                {
                  question: "For the given dataset, calculate entropy and construct decision tree:\n| Play | Weather | Temperature | Class |\n|------|---------|-------------|-------|\n| 1 | Sunny | Hot | No |\n| 2 | Sunny | Cold | No |\n| 3 | Rainy | Cold | Yes |\n| 4 | Rainy | Hot | Yes |\n(10 marks)",
                  answer: "Step 1: Calculate Entropy(S)\nTotal=4, Yes=2, No=2\nH(S) = -(2/4)log₂(2/4) - (2/4)log₂(2/4)\n     = -0.5(-1) - 0.5(-1) = 1.0\n\nStep 2: Information Gain for Weather\nSunny: {1,2} → 0 Yes, 2 No → H(Sunny) = 0\nRainy: {3,4} → 2 Yes, 0 No → H(Rainy) = 0\nGain(Weather) = 1.0 - [(2/4)×0 + (2/4)×0] = 1.0\n\nStep 3: Information Gain for Temperature\nHot: {1,4} → 1 Yes, 1 No → H(Hot) = 1.0\nCold: {2,3} → 1 Yes, 1 No → H(Cold) = 1.0\nGain(Temp) = 1.0 - [(2/4)×1.0 + (2/4)×1.0] = 0\n\nStep 4: Select Weather (Gain=1.0 > 0)\n\nFinal Decision Tree:\n        [Weather?]\n        /        \\\n     Sunny      Rainy\n       ↓          ↓\n     [NO]       [YES]",
                },
              ],
              examTips: [
                "FORMULAS TO MEMORIZE: Entropy H(S) = -Σ pᵢ log₂(pᵢ) and Gain(S,A) = H(S) - Σ(|Sᵥ|/|S|)×H(Sᵥ)",
                "Use log₂ NOT natural log. Calculator tip: log₂(x) = ln(x)/0.693",
                "Common mistake: Using count instead of proportion. Always use p = count/total in log",
                "Pure subset entropy = 0 (all same class). Maximum entropy = 1 (for 50-50 binary split)",
                "Remember: Higher Information Gain = Better attribute for splitting",
                "Always show step-by-step calculations in exams - partial marks for correct approach",
                "Draw the final tree with proper labels (Root, Internal, Leaf nodes)",
                "ID3 is GREEDY - makes locally optimal choices, may not find globally optimal tree",
              ],
            },
          },
          {
            id: "artificial-neural-networks",
            title: "Artificial Neural Networks",
            content: {
              explanation: [
                "Introduction to Neural Networks: Artificial Neural Networks (ANNs) are computing systems inspired by biological neural networks in the human brain. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections. ANNs can learn complex patterns and relationships from data.",
                "Neural Network Representations: A neural network has three types of layers: (1) Input Layer - receives input features (one neuron per feature), (2) Hidden Layer(s) - perform computations using weights and activation functions, (3) Output Layer - produces the final prediction/classification. Each connection has a weight that is learned during training. Neurons also have a bias term.",
                "Biological vs Artificial Neuron: Biological neurons receive signals through dendrites, process in cell body (soma), and transmit via axon. Artificial neurons receive inputs (x₁, x₂, ...), multiply by weights (w₁, w₂, ...), sum them up (Σwᵢxᵢ + bias), apply activation function, and produce output. The artificial neuron mimics the biological neuron's input-process-output behavior.",
                "Appropriate Problems for Neural Network Learning: ANNs work best when: (1) Input is high-dimensional (images, text, signals), (2) Target function is complex and not easily expressible as rules, (3) Large amounts of training data are available, (4) Fast evaluation time is important (after training), (5) Human interpretability is not critical. Example: Malware detection from raw bytes, network intrusion detection.",
                "Perceptron: The perceptron is the simplest neural network - a single artificial neuron. It computes: output = activation(Σwᵢxᵢ + b) where wᵢ are weights, xᵢ are inputs, b is bias. For binary classification, step function is used: output = 1 if (Σwᵢxᵢ + b) ≥ threshold, else 0. Perceptrons can implement AND, OR, NOT gates but NOT XOR (linearly separable problems only).",
                "Perceptron Learning Rule: To train a perceptron: (1) Initialize weights randomly, (2) For each training example, compute output, (3) Update weights: wᵢ_new = wᵢ_old + η(target - output)×xᵢ where η is learning rate (typically 0.1-1.0), target is desired output, output is actual output. (4) Repeat until convergence or max iterations.",
                "Basics of Backpropagation Algorithm: Backpropagation is used to train multi-layer neural networks. It has two phases: (1) Forward Pass - input propagates through network layer by layer, computing outputs, (2) Backward Pass - error is computed at output layer, then propagated backwards to compute gradients for each weight, weights are updated using gradient descent: w = w - η × ∂Error/∂w. This process repeats for many epochs until error is minimized.",
              ],
              keyPoints: [
                "ANN: Computing system inspired by biological neurons, learns from data",
                "Structure: Input Layer → Hidden Layer(s) → Output Layer",
                "Each connection has a weight; each neuron has a bias",
                "Perceptron: Simplest NN, single neuron, can solve linearly separable problems",
                "Perceptron formula: output = activation(Σwᵢxᵢ + b)",
                "Weight update rule: w_new = w_old + η(target - output) × input",
                "Perceptron limitation: Cannot solve XOR (not linearly separable)",
                "Backpropagation: Forward pass (compute output) → Backward pass (propagate error, update weights)",
                "Activation functions: Step (perceptron), Sigmoid (1/(1+e⁻ˣ)), ReLU (max(0,x))",
              ],
              diagrams: [
                {
                  title: "Perceptron Model",
                  description: "Draw a single perceptron with: Multiple inputs (x₁, x₂, x₃) on left, each connected by arrows labeled with weights (w₁, w₂, w₃) to a circle (neuron body). Inside circle show: Σ (summation) and 'f' (activation). Add bias input b. Show output arrow on right. Label formula: output = f(Σwᵢxᵢ + b). Show step function graph for activation.",
                  imageUrl: "/diagrams/perceptron_diagram.png",
                },
                {
                  title: "Multi-Layer Neural Network Architecture",
                  description: "Draw three vertical columns of circles: Input Layer (3-4 neurons labeled x₁, x₂...), Hidden Layer (4-5 neurons), Output Layer (1-2 neurons labeled 'Output'). Draw lines connecting every neuron in one layer to every neuron in next layer. Label the layers. Show that connections have weights. Add small arrows showing direction of information flow (left to right).",
                  imageUrl: "/diagrams/artificial_neural_network.svg",
                },
                {
                  title: "Biological vs Artificial Neuron Comparison",
                  description: "Draw side by side: LEFT - Biological neuron with dendrites (inputs), cell body/soma (processing), axon (output). RIGHT - Artificial neuron with input arrows (x₁, x₂), weighted connections (w₁, w₂), summation + activation circle, output arrow. Draw connecting lines showing correspondence: Dendrites ↔ Inputs, Soma ↔ Summation+Activation, Axon ↔ Output.",
                  imageUrl: "/diagrams/bio_vs_artificial_neuron.svg",
                },
                {
                  title: "Backpropagation Flow",
                  description: "Draw a 3-layer network (input-hidden-output). Show two-phase arrows: (1) Forward Pass - solid arrows flowing left to right through network, labeled 'Compute outputs'. (2) Backward Pass - dashed arrows flowing right to left, labeled 'Propagate errors, update weights'. Show error computation at output comparing with target.",
                  imageUrl: "/diagrams/backpropagation.svg",
                },
              ],
              examples: [
                {
                  title: "Implementing AND Gate using Perceptron",
                  problem: "Design a perceptron to implement the AND logic gate.",
                  solution: "AND Gate Truth Table:\nx₁ | x₂ | Output\n0  | 0  | 0\n0  | 1  | 0\n1  | 0  | 0\n1  | 1  | 1\n\nPerceptron Design:\n- Weights: w₁ = 1, w₂ = 1\n- Bias (threshold): b = -1.5\n- Activation: Step function (output = 1 if sum ≥ 0)\n\nVerification:\n- (0,0): 1(0) + 1(0) - 1.5 = -1.5 < 0 → Output = 0 ✓\n- (0,1): 1(0) + 1(1) - 1.5 = -0.5 < 0 → Output = 0 ✓\n- (1,0): 1(1) + 1(0) - 1.5 = -0.5 < 0 → Output = 0 ✓\n- (1,1): 1(1) + 1(1) - 1.5 = 0.5 ≥ 0 → Output = 1 ✓",
                },
                {
                  title: "Implementing OR Gate using Perceptron",
                  problem: "Design a perceptron to implement the OR logic gate.",
                  solution: "OR Gate Truth Table:\nx₁ | x₂ | Output\n0  | 0  | 0\n0  | 1  | 1\n1  | 0  | 1\n1  | 1  | 1\n\nPerceptron Design:\n- Weights: w₁ = 1, w₂ = 1\n- Bias (threshold): b = -0.5\n- Activation: Step function\n\nVerification:\n- (0,0): 1(0) + 1(0) - 0.5 = -0.5 < 0 → Output = 0 ✓\n- (0,1): 1(0) + 1(1) - 0.5 = 0.5 ≥ 0 → Output = 1 ✓\n- (1,0): 1(1) + 1(0) - 0.5 = 0.5 ≥ 0 → Output = 1 ✓\n- (1,1): 1(1) + 1(1) - 0.5 = 1.5 ≥ 0 → Output = 1 ✓",
                },
                {
                  title: "Why Perceptron Cannot Solve XOR",
                  problem: "Explain why a single perceptron cannot implement XOR gate.",
                  solution: "XOR Truth Table:\nx₁ | x₂ | Output\n0  | 0  | 0\n0  | 1  | 1\n1  | 0  | 1\n1  | 1  | 0\n\nThe perceptron creates a linear decision boundary (a straight line in 2D).\n\nPlotting XOR points:\n- (0,0)→0 and (1,1)→0 should be on one side\n- (0,1)→1 and (1,0)→1 should be on other side\n\nNo single straight line can separate these points!\n\nSolution: Use Multi-Layer Perceptron (MLP) with hidden layer.\nXOR can be expressed as: XOR(x₁,x₂) = AND(OR(x₁,x₂), NAND(x₁,x₂))",
                },
              ],
              problems: [
                {
                  question: "Draw and explain the structure of an artificial neuron (perceptron). (3 marks)",
                  answer: "Structure of Perceptron:\n[DIAGRAM: Show inputs x₁, x₂, x₃ connected via weights w₁, w₂, w₃ to a summation node, then activation function, then output]\n\nComponents:\n1. Inputs (x₁, x₂, ...xₙ): Feature values from data\n2. Weights (w₁, w₂, ...wₙ): Learned parameters, determine input importance\n3. Bias (b): Additional parameter for threshold adjustment\n4. Summation (Σ): Computes weighted sum = Σwᵢxᵢ + b\n5. Activation Function (f): Introduces non-linearity (step, sigmoid, ReLU)\n6. Output: f(Σwᵢxᵢ + b)\n\nThe perceptron mimics a biological neuron's input-process-output behavior.",
                },
                {
                  question: "Explain the Perceptron Learning Rule with formula. (5 marks)",
                  answer: "Perceptron Learning Rule:\n\nPurpose: To adjust weights so perceptron correctly classifies training examples.\n\nWeight Update Formula:\nw_new = w_old + η × (target - output) × input\n\nWhere:\n- w_new = updated weight\n- w_old = current weight\n- η (eta) = learning rate (typically 0.1 to 1.0)\n- target = desired output (from training data)\n- output = actual perceptron output\n- input = input value for this weight\n\nAlgorithm Steps:\n1. Initialize weights randomly (small values)\n2. For each training example:\n   a. Compute output = f(Σwᵢxᵢ + b)\n   b. Calculate error = target - output\n   c. If error ≠ 0, update each weight using formula\n3. Repeat until all examples classified correctly or max iterations\n\nKey insight: Weights only update when there's an error (target ≠ output).",
                },
                {
                  question: "Explain the basics of the Backpropagation algorithm. (5 marks)",
                  answer: "Backpropagation Algorithm:\n\nPurpose: Train multi-layer neural networks by propagating errors backward.\n\nTwo Phases:\n\n1. FORWARD PASS:\n- Input is fed to input layer\n- Each layer computes: output = activation(Σwᵢxᵢ + b)\n- Activations propagate forward through hidden layers\n- Final output is produced at output layer\n- Error is calculated: E = ½(target - output)²\n\n2. BACKWARD PASS:\n- Compute gradient of error with respect to output layer weights\n- Propagate error gradient backward through each layer\n- Use chain rule to compute gradients for hidden layer weights\n- Update all weights using gradient descent:\n  w = w - η × (∂E/∂w)\n\n3. REPEAT:\n- Process all training examples (one epoch)\n- Repeat for many epochs until error is minimized\n\nKey concepts: Gradient descent, chain rule, error minimization.",
                },
              ],
              examTips: [
                "Memorize Perceptron formula: output = f(Σwᵢxᵢ + b)",
                "Weight update rule: w_new = w_old + η(target - output) × input",
                "Know how to implement AND, OR gates - common exam question",
                "Remember: Single perceptron CANNOT solve XOR (not linearly separable)",
                "Backpropagation has 2 phases: Forward (compute output) and Backward (update weights)",
                "Draw diagrams clearly: label all inputs, weights, summation, activation, output",
                "Activation functions to know: Step (0 or 1), Sigmoid (0 to 1), ReLU (max(0,x))",
                "Multi-layer networks can solve non-linearly separable problems like XOR",
              ],
            },
          },
        ],
      },
      {
        id: "unit-2",
        title: "Bayesian, Instance-Based & Analytical Learning",
        description: "Bayes theorem, Naive Bayes classifier, k-NN, locally weighted regression, and analytical learning approaches",
        topics: [
          {
            id: "bayesian-learning",
            title: "Bayesian Learning",
            content: {
              explanation: [
                "Bayesian Learning is a probabilistic approach to machine learning that uses Bayes' theorem to update the probability of a hypothesis as more evidence becomes available. It provides a principled way to combine prior knowledge with observed data.",
                "Bayes' Theorem: P(h|D) = P(D|h) × P(h) / P(D), where P(h|D) is the posterior probability of hypothesis h given data D, P(D|h) is the likelihood, P(h) is the prior probability, and P(D) is the evidence. This formula is fundamental to all Bayesian learning.",
                "Bayes Theorem and Concept Learning: In concept learning, we want to find the most probable hypothesis given training data. The Maximum A Posteriori (MAP) hypothesis is: h_MAP = argmax P(h|D) = argmax P(D|h)P(h). If all hypotheses are equally probable, this simplifies to Maximum Likelihood (ML).",
                "Minimum Description Length (MDL) Principle: Prefer the hypothesis that minimizes the total description length = length of hypothesis + length of data given hypothesis. This provides a theoretical basis for Occam's Razor - preferring simpler hypotheses.",
                "Bayes Optimal Classifier: The classifier that minimizes the expected error by considering all hypotheses weighted by their posterior probabilities. It's optimal but often computationally intractable.",
                "Naive Bayes Classifier: A practical Bayesian classifier that assumes all features are conditionally independent given the class. Despite this 'naive' assumption, it works remarkably well for text classification, spam detection, and sentiment analysis. Formula: P(class|features) ∝ P(class) × ∏ P(feature_i|class)",
              ],
              keyPoints: [
                "Bayes' Theorem: P(h|D) = P(D|h) × P(h) / P(D)",
                "MAP hypothesis maximizes posterior probability P(h|D)",
                "ML hypothesis maximizes likelihood P(D|h) when priors are equal",
                "MDL: Prefer hypothesis that minimizes description length",
                "Naive Bayes assumes feature independence given class",
                "Naive Bayes is fast, simple, and effective for text classification",
                "Used in spam filtering, sentiment analysis, document classification",
              ],
              diagrams: [
                {
                  title: "Bayes Theorem Components",
                  description: "Draw a diagram showing: Prior P(h) + Likelihood P(D|h) → Bayes' Theorem → Posterior P(h|D). Show arrows indicating how prior belief is updated with evidence to give posterior belief. Label each component clearly.",
                  imageUrl: "/diagrams/bayes_theorem_diagram.png",
                },
                {
                  title: "Naive Bayes Classifier",
                  description: "Draw a tree diagram: Class node at top, branching down to multiple Feature nodes (F1, F2, F3...). Show that each feature is conditionally independent given the class. Write the formula: P(C|F1,F2,...) ∝ P(C) × P(F1|C) × P(F2|C) × ...",
                  imageUrl: "/diagrams/naive_bayes.svg",
                },
              ],
              examples: [
                {
                  title: "Spam Classification using Naive Bayes",
                  problem: "Given: P(Spam)=0.3, P(Ham)=0.7, P('free'|Spam)=0.8, P('free'|Ham)=0.1. What is P(Spam|'free')?",
                  solution: "Using Bayes' Theorem:\nP(Spam|'free') = P('free'|Spam) × P(Spam) / P('free')\n\nFirst, calculate P('free'):\nP('free') = P('free'|Spam)×P(Spam) + P('free'|Ham)×P(Ham)\n         = 0.8 × 0.3 + 0.1 × 0.7 = 0.24 + 0.07 = 0.31\n\nNow apply Bayes:\nP(Spam|'free') = (0.8 × 0.3) / 0.31 = 0.24 / 0.31 = 0.774\n\nResult: 77.4% probability the email is spam given it contains 'free'",
                },
              ],
              problems: [
                {
                  question: "State Bayes' Theorem and explain each component. (3 marks)",
                  answer: "Bayes' Theorem: P(h|D) = P(D|h) × P(h) / P(D)\n\nComponents:\n1. P(h|D) - Posterior: Probability of hypothesis h after seeing data D\n2. P(D|h) - Likelihood: Probability of observing data D if h is true\n3. P(h) - Prior: Initial probability of hypothesis before seeing data\n4. P(D) - Evidence: Total probability of observing the data\n\nThe theorem shows how to update our belief in a hypothesis when new evidence is observed.",
                },
                {
                  question: "Explain Naive Bayes classifier with its assumption and formula. (5 marks)",
                  answer: "Naive Bayes Classifier:\n\nAssumption: All features are conditionally independent given the class label. This means P(F1,F2|C) = P(F1|C) × P(F2|C).\n\nFormula: P(C|F1,F2,...,Fn) = P(C) × ∏P(Fi|C) / P(F1,F2,...,Fn)\n\nClassification: Assign class with highest posterior probability.\n\nAdvantages:\n1. Simple and fast to train\n2. Works well with high-dimensional data\n3. Requires small training data\n4. Handles missing values well\n\nApplications: Spam filtering, sentiment analysis, document classification, medical diagnosis.",
                },
              ],
              examTips: [
                "Memorize Bayes' Theorem formula: P(h|D) = P(D|h)P(h)/P(D)",
                "Know the difference between MAP and ML estimation",
                "Naive Bayes assumes CONDITIONAL independence, not complete independence",
                "Practice numerical problems on Bayes' theorem",
                "MDL provides theoretical justification for preferring simpler hypotheses",
              ],
            },
          },
          {
            id: "instance-based-learning",
            title: "Instance-Based Learning",
            content: {
              explanation: [
                "Instance-Based Learning (also called lazy learning) stores training examples and generalizes only when a new instance needs to be classified. Unlike eager learners that build a model during training, instance-based learners defer processing until prediction time.",
                "k-Nearest Neighbor (k-NN) Algorithm: To classify a new instance, find the k closest training examples (neighbors) and assign the majority class among them. Distance is typically measured using Euclidean distance: d(x,y) = √Σ(xi-yi)². The value of k affects bias-variance tradeoff: small k = low bias, high variance; large k = high bias, low variance.",
                "k-NN Algorithm Steps: (1) Store all training examples, (2) For new instance x, compute distance to all training examples, (3) Select k nearest neighbors, (4) Return majority class (classification) or average value (regression).",
                "Locally Weighted Regression (LWR): A non-parametric regression method that fits a weighted linear model around each query point. Points closer to the query receive higher weights. The weight function is typically Gaussian: w(i) = exp(-d(x,x_i)²/2σ²). This allows the model to adapt locally to the data.",
                "LWR constructs a local approximation for each new query rather than a single global model. This makes it more flexible but computationally expensive for large datasets.",
              ],
              keyPoints: [
                "Instance-based = Lazy learning (no model built during training)",
                "k-NN: Classify based on majority vote of k nearest neighbors",
                "Distance measure: Usually Euclidean √Σ(xi-yi)²",
                "k value: Small k = overfit, Large k = underfit",
                "Normalize features to prevent dominance by large-scale features",
                "LWR: Fits local weighted linear model for each query",
                "LWR weight: Higher for closer points, lower for distant points",
              ],
              diagrams: [
                {
                  title: "k-NN Classification",
                  description: "Draw a 2D scatter plot with two classes (circles and triangles). Mark a new query point with '?'. Draw a circle around it encompassing k=3 or k=5 nearest points. Show that classification depends on majority class within the circle. Demonstrate how different k values can give different results.",
                  imageUrl: "/diagrams/knn_classification.svg",
                },
                {
                  title: "Locally Weighted Regression",
                  description: "Draw a curve with data points. Mark a query point x. Show decreasing weights for points farther from x (larger points = higher weights near x, smaller points = lower weights far from x). Draw a local linear fit around x showing the weighted regression line.",
                  imageUrl: "/diagrams/locally_weighted_regression.svg",
                },
              ],
              examples: [
                {
                  title: "k-NN Classification Example",
                  problem: "Given training data: (1,1)→A, (2,1)→A, (4,3)→B, (5,4)→B, (3,2)→B. Classify new point (2,2) using k=3.",
                  solution: "Step 1: Calculate distances from (2,2) to all points:\n- d((2,2),(1,1)) = √((2-1)² + (2-1)²) = √2 ≈ 1.41\n- d((2,2),(2,1)) = √((2-2)² + (2-1)²) = 1\n- d((2,2),(4,3)) = √((2-4)² + (2-3)²) = √5 ≈ 2.24\n- d((2,2),(5,4)) = √((2-5)² + (2-4)²) = √13 ≈ 3.61\n- d((2,2),(3,2)) = √((2-3)² + (2-2)²) = 1\n\nStep 2: Find k=3 nearest neighbors:\n(2,1)→A (dist=1), (3,2)→B (dist=1), (1,1)→A (dist=1.41)\n\nStep 3: Majority vote: 2 A's, 1 B\n\nResult: Classify (2,2) as class A",
                },
              ],
              problems: [
                {
                  question: "Explain the k-NN algorithm with steps. What is the effect of k value? (5 marks)",
                  answer: "k-Nearest Neighbor Algorithm:\n\nSteps:\n1. Store all training examples\n2. For a new query point x:\n   a. Calculate distance from x to all training examples\n   b. Select k nearest neighbors\n   c. For classification: Return majority class\n   d. For regression: Return average of neighbor values\n\nEffect of k value:\n- Small k (e.g., k=1): Low bias, high variance. Model is sensitive to noise and outliers. Risk of overfitting.\n- Large k: High bias, low variance. Model is smoother but may miss local patterns. Risk of underfitting.\n\nTypically, k is chosen using cross-validation. Common practice: use odd k for binary classification to avoid ties.",
                },
                {
                  question: "Explain Locally Weighted Regression. How is it different from standard linear regression? (5 marks)",
                  answer: "Locally Weighted Regression (LWR):\n\nConcept: Instead of fitting one global linear model, LWR fits a weighted linear model around each query point, giving higher weights to nearby training examples.\n\nWeight Function: w(i) = exp(-d(x,x_i)²/2σ²)\n- Points close to query x get weight ≈ 1\n- Points far from x get weight ≈ 0\n\nDifferences from Standard Linear Regression:\n1. Standard LR: One global model, constant coefficients\n   LWR: Local model for each query, varying coefficients\n2. Standard LR: Equal weight to all points\n   LWR: Distance-based weights\n3. Standard LR: Fast prediction, fixed model\n   LWR: Slower prediction, adaptive model\n\nAdvantage: Can model complex non-linear relationships.\nDisadvantage: Computationally expensive for large datasets.",
                },
              ],
              examTips: [
                "Know k-NN steps: Store → Compute distances → Find k nearest → Vote",
                "Remember: k-NN is LAZY learning (no training phase)",
                "Euclidean distance formula: √Σ(xi-yi)²",
                "LWR uses Gaussian kernel for distance-based weighting",
                "Practice numerical problems on k-NN classification",
              ],
            },
          },
          {
            id: "analytical-learning",
            title: "Analytical Learning",
            content: {
              explanation: [
                "Analytical Learning uses prior knowledge and logical reasoning to analyze training examples rather than just inductively generalizing from them. It combines the power of deductive reasoning with learning from examples.",
                "Inductive vs Analytical Learning: Inductive learning generalizes from specific examples to general rules without using prior knowledge. Analytical learning uses prior knowledge (domain theory) to explain and generalize from examples. Analytical methods require fewer examples but need accurate domain knowledge.",
                "Explanation-Based Generalization (EBG): A technique that uses domain theory to construct an explanation of why a training example is a member of the target concept. Then it generalizes this explanation to form a general rule. The goal is to extract the relevant features based on the explanation.",
                "PROLOG-EBG: An implementation of EBG using PROLOG (a logic programming language). Given a positive training example, domain theory, and operationality criterion (what features can be used), PROLOG-EBG: (1) Explains why the example satisfies the goal, (2) Analyzes the explanation to find relevant features, (3) Generalizes to create a rule.",
                "EBG Steps: (1) EXPLAIN: Construct proof that example satisfies target concept using domain theory, (2) ANALYZE: Identify the most general conditions under which the explanation holds, (3) REFINE: Create a new rule that generalizes the explanation.",
              ],
              keyPoints: [
                "Analytical learning uses prior knowledge + examples",
                "Inductive: Examples → General rules (no prior knowledge)",
                "Analytical: Prior knowledge + Examples → Justified rules",
                "EBG: Explain → Analyze → Generalize",
                "Requires accurate domain theory",
                "Fewer examples needed compared to inductive learning",
                "PROLOG-EBG implements EBG in logic programming",
              ],
              diagrams: [
                {
                  title: "Inductive vs Analytical Learning",
                  description: "Draw two parallel flowcharts. LEFT (Inductive): Multiple Examples → Pattern Recognition → General Hypothesis. RIGHT (Analytical): Prior Knowledge (Domain Theory) + Example → Explanation → Generalized Rule. Show that analytical uses knowledge to guide generalization.",
                  imageUrl: "/diagrams/inductive_vs_analytical.svg",
                },
                {
                  title: "EBG Process",
                  description: "Draw three connected boxes: (1) EXPLAIN - Show domain theory + example leading to explanation/proof, (2) ANALYZE - Show extracting relevant features from explanation, (3) REFINE - Show creating generalized rule. Use arrows to show the flow.",
                  imageUrl: "/diagrams/ebg_process.svg",
                },
              ],
              problems: [
                {
                  question: "Compare Inductive and Analytical Learning. (5 marks)",
                  answer: "Inductive Learning:\n- Generalizes from specific examples to general rules\n- No prior domain knowledge required\n- Requires many training examples\n- May learn incorrect hypotheses from noisy data\n- Example: Decision trees, Neural networks\n\nAnalytical Learning:\n- Uses prior domain theory to analyze examples\n- Requires accurate domain knowledge\n- Can learn from very few examples\n- Extracts relevant features based on explanation\n- Example: Explanation-Based Generalization (EBG)\n\nKey Difference: Inductive learning treats examples as primary source; Analytical learning uses domain theory as primary source with examples for refinement.",
                },
                {
                  question: "Explain Explanation-Based Generalization (EBG) with its steps. (5 marks)",
                  answer: "Explanation-Based Generalization (EBG):\n\nPurpose: Use domain theory to generalize from a single example by constructing an explanation.\n\nInputs Required:\n1. Target concept to learn\n2. Training example (positive instance)\n3. Domain theory (background knowledge)\n4. Operationality criterion (usable features)\n\nSteps:\n1. EXPLAIN: Construct a proof/explanation showing why the example satisfies the target concept using domain theory.\n2. ANALYZE: Determine the most general conditions under which the explanation holds. Identify relevant features.\n3. REFINE: Create a new Horn clause rule that captures the generalized explanation.\n\nAdvantage: Can learn from a single example.\nLimitation: Requires accurate and complete domain theory.",
                },
              ],
              examTips: [
                "Key difference: Inductive uses examples only, Analytical uses prior knowledge",
                "EBG steps: EXPLAIN → ANALYZE → REFINE (or GENERALIZE)",
                "Analytical learning needs FEWER examples but requires domain theory",
                "PROLOG-EBG uses Horn clauses for knowledge representation",
                "Remember: EBG can learn from a SINGLE example if domain theory is good",
              ],
            },
          },
        ],
      },
      {
        id: "unit-3",
        title: "Combined Learning & Reinforcement Learning",
        description: "Combining inductive and analytical approaches, introduction to reinforcement learning and Q-learning",
        topics: [
          {
            id: "combined-learning",
            title: "Combining Inductive and Analytical Learning",
            content: {
              explanation: [
                "Motivation: Pure inductive learning requires many examples and ignores available domain knowledge. Pure analytical learning requires perfect domain theory. Combining both approaches leverages the strengths of each: use domain knowledge when available, fall back on inductive learning when knowledge is incomplete.",
                "Inductive-Analytical Approaches combine the generalization power of inductive learning with the knowledge-guided focus of analytical learning. Examples include KBANN (Knowledge-Based Artificial Neural Networks) and FOCL (First Order Combined Learner).",
                "KBANN (Knowledge-Based Artificial Neural Networks): Initializes a neural network with prior knowledge encoded as rules. The network structure is derived from the domain theory, and then weights are refined using standard backpropagation with training data. This combines the knowledge-representation of rules with the learning capability of neural networks.",
                "Using Prior Knowledge to Initialize Hypotheses: Instead of starting from random or empty hypotheses, use domain theory to create an initial hypothesis. Then refine this hypothesis using training data through inductive methods. This speeds up learning and often leads to better final hypotheses.",
                "Benefits of combining: (1) Faster learning with fewer examples, (2) Better handling of incomplete or incorrect domain knowledge, (3) More interpretable results when rules are extracted.",
              ],
              keyPoints: [
                "Motivation: Neither pure inductive nor pure analytical is ideal",
                "Combined approaches use domain knowledge + examples",
                "KBANN: Initialize neural network from domain rules, then train",
                "FOCL: Combines first-order logic with inductive learning",
                "Prior knowledge provides better starting hypothesis",
                "Inductive refinement handles errors in domain theory",
                "Result: Faster learning, fewer examples needed",
              ],
              diagrams: [
                {
                  title: "Combined Learning Approach",
                  description: "Draw a diagram showing: Domain Theory → Initial Hypothesis → Combined with Training Examples → Inductive Refinement → Final Hypothesis. Show that prior knowledge provides the starting point and inductive learning refines it.",
                  imageUrl: "/diagrams/inductive_learning.svg",
                },
                {
                  title: "KBANN Architecture",
                  description: "Draw a neural network where the initial connections are derived from IF-THEN rules in domain theory. Show: Rules → Network Structure → Backpropagation Training with Examples → Refined Network. Label that network topology comes from knowledge.",
                  imageUrl: "/diagrams/kbann.svg",
                },
              ],
              problems: [
                {
                  question: "What is the motivation for combining inductive and analytical learning? (3 marks)",
                  answer: "Motivation for Combining Inductive and Analytical Learning:\n\n1. Pure Inductive Learning Limitations:\n   - Requires large number of training examples\n   - Ignores available domain knowledge\n   - May learn incorrect hypotheses\n\n2. Pure Analytical Learning Limitations:\n   - Requires perfect, complete domain theory\n   - Cannot handle noise in data\n   - Failed if domain theory is incomplete\n\n3. Solution - Combined Approach:\n   - Uses domain knowledge when available\n   - Falls back on induction when knowledge is incomplete\n   - Requires fewer examples than pure induction\n   - Tolerates imperfect domain theory",
                },
                {
                  question: "Explain how prior knowledge can be used to initialize the hypothesis in learning. (5 marks)",
                  answer: "Using Prior Knowledge to Initialize Hypothesis:\n\nApproach:\nInstead of starting learning from scratch (empty/random hypothesis), use domain knowledge to create a good initial hypothesis, then refine with training data.\n\nMethods:\n1. KBANN (Knowledge-Based ANN):\n   - Convert domain rules to neural network structure\n   - Set initial weights based on rules\n   - Refine using backpropagation\n\n2. FOCL (First Order Combined Learner):\n   - Start with first-order logic rules\n   - Use inductive learning to refine/extend rules\n\nBenefits:\n1. Faster convergence during training\n2. Better generalization with fewer examples\n3. Handles cases where domain theory is approximate\n4. More interpretable final model\n\nLimitation: Requires some prior domain knowledge to exist.",
                },
              ],
              examTips: [
                "Remember: Combined = Best of both inductive and analytical",
                "KBANN = Rules → Neural Network → Backprop training",
                "Key benefit: Can handle INCOMPLETE domain knowledge",
                "Prior knowledge speeds up learning and improves accuracy",
              ],
            },
          },
          {
            id: "reinforcement-learning",
            title: "Reinforcement Learning",
            content: {
              explanation: [
                "Introduction to Reinforcement Learning (RL): Unlike supervised learning (learns from labeled examples) and unsupervised learning (finds patterns), RL learns from interaction with an environment through trial and error. The agent takes actions, receives rewards/penalties, and learns to maximize cumulative reward.",
                "Key Components: (1) Agent - the learner/decision maker, (2) Environment - what agent interacts with, (3) State (s) - current situation, (4) Action (a) - choices available to agent, (5) Reward (r) - feedback signal, (6) Policy (π) - mapping from states to actions.",
                "The Learning Task: The agent must learn a policy π(s)→a that maximizes the expected cumulative reward over time. The challenge: rewards may be delayed (action now, reward later), and the agent must balance exploration (trying new actions) vs exploitation (using known good actions).",
                "Q-Learning: A model-free RL algorithm that learns the value of state-action pairs. Q(s,a) represents the expected future reward for taking action a in state s and following optimal policy thereafter. No model of environment needed - learns directly from experience.",
                "Q-Learning Update Rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)], where α is learning rate (0-1), γ is discount factor (0-1) for future rewards, r is immediate reward, s' is new state, and max Q(s',a') is best expected future value.",
                "Q-Learning Algorithm: (1) Initialize Q-table with zeros, (2) For each episode: observe state s, choose action (ε-greedy), take action, observe reward r and new state s', update Q(s,a), repeat until terminal state. Over time, Q-values converge to optimal values.",
              ],
              keyPoints: [
                "RL: Agent learns by interacting with environment",
                "Key concepts: State, Action, Reward, Policy",
                "Goal: Maximize cumulative reward over time",
                "Explore vs Exploit: Try new things vs use known good actions",
                "Q-Learning: Learn Q(s,a) = value of action a in state s",
                "Q-Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]",
                "α = learning rate, γ = discount factor",
                "Model-free: Doesn't need to know environment dynamics",
              ],
              diagrams: [
                {
                  title: "Reinforcement Learning Framework",
                  description: "Draw a cycle diagram: Agent → takes Action → Environment → returns State and Reward → Agent. Show arrows forming a loop. Label: Agent observes state, selects action, environment provides next state and reward.",
                  imageUrl: "/diagrams/qlearning_diagram.png",
                },
                {
                  title: "Q-Learning Process",
                  description: "Draw a grid world (simple 3x3 or 4x4 grid). Mark Start (S) and Goal (G) states. Show Q-values in cells as a table. Illustrate agent moving, receiving reward at goal, and Q-values being updated. Show the update formula: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]",
                  imageUrl: "/diagrams/qlearning_process.svg",
                },
              ],
              examples: [
                {
                  title: "Q-Learning Update Example",
                  problem: "Current Q(s1, go_right) = 0, agent takes action 'go_right' from s1, receives reward r=10, reaches s2 where max Q(s2,a) = 5. Given α=0.5, γ=0.9. Calculate new Q(s1, go_right).",
                  solution: "Using Q-Learning update rule:\nQ(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]\n\nSubstituting values:\nQ(s1, go_right) ← 0 + 0.5[10 + 0.9(5) - 0]\nQ(s1, go_right) ← 0 + 0.5[10 + 4.5 - 0]\nQ(s1, go_right) ← 0 + 0.5[14.5]\nQ(s1, go_right) ← 7.25\n\nNew Q-value: Q(s1, go_right) = 7.25",
                },
              ],
              problems: [
                {
                  question: "Explain the reinforcement learning framework with its key components. (5 marks)",
                  answer: "Reinforcement Learning Framework:\n\nDefinition: Agent learns optimal behavior through trial-and-error interaction with environment, receiving rewards as feedback.\n\nKey Components:\n1. Agent: The learner/decision-maker\n2. Environment: External system agent interacts with\n3. State (s): Current situation/configuration\n4. Action (a): Possible moves agent can make\n5. Reward (r): Numerical feedback (+ve or -ve)\n6. Policy (π): Strategy mapping states to actions\n\nLearning Loop:\n1. Agent observes current state s\n2. Agent selects action a based on policy\n3. Environment transitions to new state s'\n4. Environment provides reward r\n5. Agent updates policy based on experience\n6. Repeat\n\nGoal: Learn policy that maximizes cumulative reward.",
                },
                {
                  question: "Explain Q-Learning algorithm with its update rule. (5 marks)",
                  answer: "Q-Learning Algorithm:\n\nConcept: Model-free RL algorithm that learns Q-values representing expected utility of state-action pairs.\n\nQ-Value: Q(s,a) = expected cumulative reward for taking action a in state s and following optimal policy thereafter.\n\nUpdate Rule:\nQ(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]\n\nWhere:\n- α (alpha): Learning rate (0 to 1), controls update speed\n- γ (gamma): Discount factor (0 to 1), importance of future rewards\n- r: Immediate reward received\n- s': New state after action\n- max Q(s',a'): Best Q-value from new state\n\nAlgorithm Steps:\n1. Initialize Q-table with zeros\n2. For each episode:\n   a. Start in initial state s\n   b. Choose action a (ε-greedy)\n   c. Execute action, observe r and s'\n   d. Update Q(s,a) using formula\n   e. s ← s'\n   f. Repeat until terminal state\n\nProperties: Guaranteed to converge to optimal Q-values.",
                },
              ],
              examTips: [
                "Memorize Q-Update: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]",
                "Know components: α = learning rate, γ = discount factor",
                "RL differs from supervised (no labels) and unsupervised (has rewards)",
                "Explore vs Exploit: ε-greedy balances both",
                "Q-Learning is MODEL-FREE (doesn't need environment model)",
                "Practice numerical problems on Q-value updates",
              ],
            },
          },
        ],
      },
    ],
  },
  "firewall-utm": {
    id: "firewall-utm",
    title: "Firewall and UTM Architecture",
    shortTitle: "Firewall & UTM",
    description:
      "Comprehensive study of firewall technologies, unified threat management, and network security architecture design.",
    units: [
      {
        id: "unit-1",
        title: "Introduction to Network Security & Firewalls",
        description: "Computer and network security concepts, principles, and history of firewalls",
        topics: [
          {
            id: "network-security-concepts",
            title: "Computer and Network Security Concepts",
            content: {
              explanation: [
                "Computer Security refers to the protection of computer systems and information from unauthorized access, theft, damage, or disruption. It encompasses hardware, software, and data protection through various technical and administrative controls.",
                "Network Security focuses on protecting the integrity, confidentiality, and availability of data as it is transmitted across or accessed through networks. It involves policies, practices, and technologies to prevent and monitor unauthorized access, misuse, modification, or denial of the network.",
                "Key Security Principles (CIA Triad): (1) Confidentiality - ensuring information is accessible only to authorized users, (2) Integrity - maintaining accuracy and completeness of data, (3) Availability - ensuring authorized users have access when needed. Additional principles include Authentication, Non-repudiation, and Accountability.",
                "Security Threats include: Malware (viruses, worms, trojans, ransomware), Network attacks (DoS, DDoS, Man-in-the-middle), Social engineering (phishing, pretexting), Unauthorized access (hacking, privilege escalation), and Insider threats.",
                "Defense-in-Depth Strategy uses multiple layers of security controls: Physical security, Network security (firewalls, IDS/IPS), Host security (antivirus, HIDS), Application security, and Data security (encryption). If one layer fails, others provide protection.",
              ],
              keyPoints: [
                "CIA Triad: Confidentiality, Integrity, Availability",
                "Additional principles: Authentication, Non-repudiation, Accountability",
                "Defense-in-Depth: Multiple layers of security controls",
                "Common threats: Malware, DoS, Man-in-the-middle, Phishing",
                "Security is a continuous process, not a one-time implementation",
                "Risk Management: Identify, Assess, Mitigate, Monitor",
              ],
              diagrams: [
                {
                  title: "CIA Triad",
                  description: "Draw a triangle with three vertices labeled: Confidentiality (top), Integrity (bottom-left), Availability (bottom-right). In the center write 'Information Security'. Add brief descriptions for each: Confidentiality = 'Who can access?', Integrity = 'Is data accurate?', Availability = 'Can authorized users access it?'",
                  imageUrl: "/diagrams/cia_triad_diagram.png",
                },
                {
                  title: "Defense-in-Depth Architecture",
                  description: "Draw concentric circles or layers: Outermost = Physical Security, then Network Security (Firewalls), then Host Security (Antivirus), then Application Security, innermost = Data Security (Encryption). Show that an attacker must breach multiple layers.",
                  imageUrl: "/diagrams/defense_in_depth.svg",
                },
              ],
              problems: [
                {
                  question: "Explain the CIA Triad with examples. (5 marks)",
                  answer: "CIA Triad - Three pillars of Information Security:\n\n1. CONFIDENTIALITY:\nEnsuring information is accessible only to authorized users.\nExamples:\n- Encryption of sensitive data\n- Access control lists (ACLs)\n- Password protection\n\n2. INTEGRITY:\nMaintaining accuracy and completeness of data.\nExamples:\n- Hashing (MD5, SHA) to detect tampering\n- Digital signatures\n- Version control systems\n\n3. AVAILABILITY:\nEnsuring authorized users can access information when needed.\nExamples:\n- Redundant systems (RAID, clustering)\n- DDoS protection\n- Backup and disaster recovery",
                },
                {
                  question: "What is Defense-in-Depth? List its layers. (3 marks)",
                  answer: "Defense-in-Depth:\nA security strategy that uses multiple layers of defense so that if one layer fails, others continue to provide protection.\n\nLayers:\n1. Physical Security - locks, guards, CCTV\n2. Perimeter Security - firewalls, DMZ\n3. Network Security - IDS/IPS, segmentation\n4. Host Security - antivirus, HIDS, patching\n5. Application Security - input validation, WAF\n6. Data Security - encryption, DLP\n7. User Security - training, policies",
                },
              ],
              examTips: [
                "Memorize CIA Triad with examples for each",
                "Know the difference between threats, vulnerabilities, and risks",
                "Defense-in-Depth: Multiple layers, no single point of failure",
                "Common exam question: Compare different security principles",
              ],
            },
          },
          {
            id: "firewall-history",
            title: "History of Firewalls",
            content: {
              explanation: [
                "Generation 1 - Packet Filters (1988): The first firewalls were simple packet filters that examined network packets and allowed/denied based on source/destination IP addresses and ports. They were stateless, treating each packet independently without understanding connections.",
                "Generation 2 - Stateful Inspection (1990s): These firewalls maintain a state table tracking active connections. They understand that a packet is part of an established connection and can make more intelligent decisions. Significantly more secure than packet filters.",
                "Generation 3 - Application Layer/Proxy Firewalls: These operate at Layer 7 (Application), understanding protocols like HTTP, FTP, SMTP. They can inspect packet content, block specific URLs or file types, and provide deeper security but at higher performance cost.",
                "Generation 4 - Next-Generation Firewalls (NGFW) (2010s): Combine traditional firewall with IPS, application awareness, user identity, SSL inspection, and threat intelligence. They can identify and control applications regardless of port/protocol.",
                "The evolution shows increasing intelligence: from simple packet filtering to understanding applications, users, and threats. Modern NGFWs are central to enterprise security, integrating multiple security functions.",
              ],
              keyPoints: [
                "Gen 1: Packet filters - stateless, IP/port based",
                "Gen 2: Stateful inspection - tracks connections",
                "Gen 3: Application layer - deep packet inspection",
                "Gen 4: NGFW - combined with IPS, app awareness, user identity",
                "Evolution: Speed vs Security tradeoff at each generation",
                "Modern trend: Integration of multiple security functions",
              ],
              diagrams: [
                {
                  title: "Firewall Evolution Timeline",
                  description: "Draw a timeline from 1988 to present. Mark: 1988 - Packet Filters, 1990s - Stateful Inspection, 2000s - Application Layer, 2010s - Next-Gen Firewalls. Show increasing complexity and capabilities at each stage.",
                  imageUrl: "/diagrams/firewall_generations_diagram.png",
                },
                {
                  title: "Firewall Generations Comparison",
                  description: "Draw a table or stacked diagram comparing: Gen 1 (Layer 3-4, IP/Port), Gen 2 (Connection tracking), Gen 3 (Deep packet inspection), Gen 4 (App awareness + IPS + User ID). Show that each generation adds capabilities on top of previous.",
                  imageUrl: "/diagrams/firewall_generations.svg",
                },
              ],
              problems: [
                {
                  question: "Explain the evolution of firewalls from packet filters to next-generation firewalls. (5 marks)",
                  answer: "Evolution of Firewalls:\n\n1. PACKET FILTERS (1988):\n- First generation, stateless\n- Filter based on IP, port, protocol\n- Fast but limited security\n- No connection awareness\n\n2. STATEFUL INSPECTION (1990s):\n- Tracks connection state\n- Allows return traffic for established connections\n- More secure than packet filters\n- Maintains state table\n\n3. APPLICATION LAYER (2000s):\n- Operates at Layer 7\n- Deep packet inspection\n- Understands HTTP, FTP, etc.\n- Can filter based on content\n\n4. NEXT-GENERATION (2010s):\n- Combines firewall + IPS + App awareness\n- User identity integration\n- SSL/TLS inspection\n- Threat intelligence feeds\n- Cloud integration",
                },
              ],
              examTips: [
                "Know the 4 generations and their key characteristics",
                "Understand what each generation adds over previous",
                "NGFW = Traditional firewall + IPS + Application control + User ID",
                "Compare performance vs security at each generation",
              ],
            },
          },
        ],
      },
      {
        id: "unit-2",
        title: "UTM Foundations",
        description: "Stateful and stateless firewalls, unified threat management foundations",
        topics: [
          {
            id: "stateful-stateless",
            title: "Stateful and Stateless Firewalls",
            content: {
              explanation: [
                "Stateless Firewall (Packet Filtering): Examines each packet independently based on predefined rules. Checks source/destination IP, port numbers, and protocol. Each packet is evaluated in isolation without any memory of previous packets. Fast but cannot understand connections or sessions.",
                "Stateful Firewall: Maintains a state table that tracks the state of active connections (TCP handshake, established, closing). It understands that packets belong to existing connections and can make context-aware decisions. Return traffic for established connections is automatically allowed.",
                "State Table tracks: Source IP, Destination IP, Source Port, Destination Port, Protocol, Connection State (NEW, ESTABLISHED, RELATED, INVALID), Timeout. When a packet arrives, firewall checks if it matches an existing connection in the state table.",
                "Comparison: Stateless firewalls are faster and use less memory but are less secure. They cannot handle protocols that negotiate ports (like FTP) without additional configuration. Stateful firewalls provide better security and ease of configuration but require more resources.",
                "Modern firewalls are stateful by default. Stateless packet filtering is still used in high-performance scenarios like hardware routers or as a first-line filter before stateful inspection.",
              ],
              keyPoints: [
                "Stateless: Each packet examined independently, no connection memory",
                "Stateful: Tracks connection state, understands sessions",
                "State table: Records active connections and their status",
                "Stateless = Fast, less secure; Stateful = Slower, more secure",
                "TCP states tracked: NEW, ESTABLISHED, RELATED, INVALID",
                "Stateful handles complex protocols (FTP active mode) better",
              ],
              diagrams: [
                {
                  title: "Stateless vs Stateful Firewall",
                  description: "Draw two diagrams side by side. LEFT (Stateless): Show packets arriving, each checked against rules independently, no memory. RIGHT (Stateful): Show packets, a state table with connection entries, and packets being matched to existing connections. Highlight that stateful remembers previous packets.",
                  imageUrl: "/diagrams/stateful_stateless_diagram.png",
                },
                {
                  title: "State Table Structure",
                  description: "Draw a table with columns: Source IP | Dest IP | Src Port | Dst Port | Protocol | State | Timeout. Show example entries: 192.168.1.10 | 8.8.8.8 | 50234 | 443 | TCP | ESTABLISHED | 3600s. Explain each column's purpose.",
                  imageUrl: "/diagrams/state_table.svg",
                },
              ],
              examples: [
                {
                  title: "TCP Connection Tracking",
                  problem: "Explain how a stateful firewall handles a TCP connection from client (192.168.1.10:50000) to server (8.8.8.8:443).",
                  solution: "1. Client sends SYN packet\n   - Firewall checks rules, allows outbound HTTPS\n   - Creates state table entry: State = NEW\n\n2. Server responds with SYN-ACK\n   - Firewall matches to state table entry\n   - Updates state to ESTABLISHED\n   - Allows packet without rule check\n\n3. Client sends ACK, connection established\n   - All subsequent packets matched to state\n   - Allowed automatically\n\n4. Connection closes (FIN/ACK)\n   - State changes to CLOSING\n   - Entry removed after timeout",
                },
              ],
              problems: [
                {
                  question: "Differentiate between stateful and stateless firewalls. (5 marks)",
                  answer: "Stateless Firewall:\n- Examines each packet independently\n- No memory of previous packets\n- Rules based on IP, port, protocol only\n- Fast, low resource usage\n- Cannot track connections\n- Requires rules for both directions\n\nStateful Firewall:\n- Tracks connection state\n- Maintains state table\n- Understands sessions and context\n- Slower, needs more memory\n- Return traffic auto-allowed\n- Better handling of complex protocols\n\nKey Difference: Stateful understands CONNECTION context; Stateless treats each PACKET independently.",
                },
                {
                  question: "What is a state table? What information does it contain? (3 marks)",
                  answer: "State Table:\nA data structure maintained by stateful firewalls to track active connections.\n\nContains:\n1. Source IP Address\n2. Destination IP Address\n3. Source Port\n4. Destination Port\n5. Protocol (TCP/UDP)\n6. Connection State (NEW, ESTABLISHED, RELATED, INVALID)\n7. Timeout value\n8. Sequence numbers (for TCP)\n\nPurpose: Allows firewall to match incoming packets to existing connections and make context-aware decisions.",
                },
              ],
              examTips: [
                "Key difference: Stateful tracks CONNECTIONS, Stateless checks PACKETS",
                "State table contains: IPs, Ports, Protocol, State, Timeout",
                "Stateful automatically allows return traffic for established connections",
                "Stateless is faster but less secure",
              ],
            },
          },
          {
            id: "utm-foundations",
            title: "Unified Threat Management Foundations",
            content: {
              explanation: [
                "Unified Threat Management (UTM) is a comprehensive security solution that consolidates multiple security functions into a single hardware or software platform. It evolved from the need to simplify security management and reduce the complexity of managing multiple separate security appliances.",
                "Core UTM Components: (1) Firewall - stateful inspection and access control, (2) IPS/IDS - intrusion prevention/detection, (3) Antivirus/Antimalware - gateway-level malware scanning, (4) VPN - secure remote access, (5) Web Filtering - URL and content filtering, (6) Email Security - spam and phishing protection.",
                "Benefits of UTM: (1) Simplified management - single console for all security, (2) Reduced cost - one device vs multiple appliances, (3) Easier deployment - single device to install and configure, (4) Integrated logging - centralized security logs, (5) Better visibility - unified view of network security.",
                "Limitations of UTM: (1) Single point of failure - if UTM fails, all security functions lost, (2) Performance bottleneck - all traffic through one device, (3) Jack of all trades - individual functions may not match dedicated appliances, (4) Scalability challenges - may not suit large enterprises.",
                "UTM is ideal for small to medium businesses (SMBs) that need comprehensive security without dedicated security staff. Large enterprises often prefer best-of-breed separate solutions for each function.",
              ],
              keyPoints: [
                "UTM = Multiple security functions in one device",
                "Components: Firewall, IPS, Antivirus, VPN, Web Filter, Email Security",
                "Benefits: Simplified management, lower cost, single console",
                "Limitations: Single point of failure, performance bottleneck",
                "Target market: Small to Medium Businesses (SMBs)",
                "Alternative: Best-of-breed (separate specialized devices)",
              ],
              diagrams: [
                {
                  title: "UTM Architecture",
                  description: "Draw a single box labeled 'UTM Device' containing multiple smaller boxes inside: Firewall, IPS, Antivirus, VPN, Web Filter, Email Security, Application Control. Show Internet on one side, Internal Network on other side, with all traffic passing through the UTM.",
                  imageUrl: "/diagrams/utm_architecture_diagram.png",
                },
                {
                  title: "UTM vs Separate Appliances",
                  description: "Draw two diagrams. TOP: Multiple separate devices (Firewall box, IPS box, Antivirus box, VPN box) connected in chain. BOTTOM: Single UTM box containing all functions. Label trade-offs: Separate = More complex, higher cost, best performance; UTM = Simple, lower cost, potential bottleneck.",
                  imageUrl: "/diagrams/utm_vs_separate.svg",
                },
              ],
              problems: [
                {
                  question: "What is Unified Threat Management? List its core components. (5 marks)",
                  answer: "Unified Threat Management (UTM):\n\nDefinition: A comprehensive security solution that consolidates multiple security functions into a single device or platform for simplified management.\n\nCore Components:\n1. Firewall - Stateful packet inspection, access control\n2. IPS/IDS - Intrusion prevention and detection\n3. Antivirus Gateway - Malware scanning at network level\n4. VPN - Site-to-site and remote access VPN\n5. Web Filtering - URL blocking, content filtering\n6. Email Security - Anti-spam, anti-phishing\n7. Application Control - Block/allow specific applications\n\nBenefits: Single management console, reduced complexity, lower total cost of ownership.",
                },
                {
                  question: "What are the advantages and disadvantages of UTM? (5 marks)",
                  answer: "Advantages of UTM:\n1. Simplified management - single console\n2. Lower cost than multiple appliances\n3. Easier deployment and configuration\n4. Integrated logging and reporting\n5. Reduced complexity\n6. Vendor support from single source\n\nDisadvantages of UTM:\n1. Single point of failure\n2. Performance bottleneck under heavy load\n3. Individual functions may not match specialized devices\n4. Scalability limitations\n5. All features may not be used (bundled)\n6. May not suit large enterprise needs\n\nBest suited for: Small to Medium Businesses (SMBs)",
                },
              ],
              examTips: [
                "Know all UTM components: FW, IPS, AV, VPN, Web Filter, Email",
                "Remember: UTM = Simplified but potential single point of failure",
                "Compare with 'best-of-breed' approach for large enterprises",
                "Common question: Benefits vs limitations of UTM",
              ],
            },
          },
        ],
      },
      {
        id: "unit-3",
        title: "UTM Concepts & Best Practices",
        description: "UTM history, comparison with other architectures, next-gen firewalls, and deployment best practices",
        topics: [
          {
            id: "utm-history-concepts",
            title: "History of UTM and Security Architectures",
            content: {
              explanation: [
                "History of UTM: The term 'Unified Threat Management' was coined by IDC (International Data Corporation) in 2004. It emerged as organizations struggled with managing multiple security point products. Early UTM devices combined firewall, VPN, and basic IPS. Over time, more features were added.",
                "UTM vs Traditional Firewalls: Traditional firewalls only perform packet filtering and stateful inspection. UTM adds IPS, antivirus, web filtering, and more. UTM provides comprehensive protection but at potential performance cost.",
                "UTM vs Next-Generation Firewalls (NGFW): While UTM focuses on consolidating multiple security functions, NGFWs focus on deep visibility and control. Key NGFW features: Application awareness and control, User identity integration, SSL/TLS inspection, Integrated threat intelligence. NGFWs are often considered enterprise-grade while UTM targets SMBs.",
                "Comparison: UTM = Breadth of features, simpler management, SMB focus. NGFW = Depth of inspection, application control, enterprise focus. Many vendors now blur these lines, offering unified solutions.",
                "Security architecture choice depends on: Organization size, budget, in-house expertise, performance requirements, and specific security needs.",
              ],
              keyPoints: [
                "UTM term coined 2004 by IDC",
                "UTM adds multiple functions to traditional firewall",
                "NGFW focuses on application awareness and user identity",
                "UTM = Breadth (many features); NGFW = Depth (deep inspection)",
                "Modern solutions often combine UTM + NGFW features",
                "Choice depends on organization size and needs",
              ],
              diagrams: [
                {
                  title: "UTM vs NGFW Comparison",
                  description: "Draw a comparison table or Venn diagram. UTM side: Multiple security functions (FW, IPS, AV, VPN, Web Filter). NGFW side: Application awareness, User identity, SSL inspection, Threat intelligence. Overlap: Firewall, IPS. Label: UTM = SMB focused, NGFW = Enterprise focused.",
                  imageUrl: "/diagrams/utm_vs_ngfw.svg",
                },
              ],
              problems: [
                {
                  question: "Compare UTM with Next-Generation Firewalls (NGFW). (5 marks)",
                  answer: "UTM (Unified Threat Management):\n- Consolidates multiple security functions\n- Firewall + IPS + AV + VPN + Web Filter\n- Focus on simplified management\n- Target market: SMBs\n- Breadth of features\n\nNGFW (Next-Generation Firewall):\n- Deep packet inspection\n- Application awareness and control\n- User identity integration\n- SSL/TLS traffic inspection\n- Integrated threat intelligence\n- Target market: Enterprises\n- Depth of inspection\n\nKey Differences:\n1. UTM focuses on feature consolidation; NGFW on deep visibility\n2. UTM for SMBs; NGFW for enterprises\n3. NGFW has stronger application control\n4. Modern solutions often combine both approaches",
                },
              ],
              examTips: [
                "Remember: UTM (2004) = consolidation; NGFW = application awareness",
                "UTM for SMBs, NGFW for enterprises",
                "Know key NGFW features: App awareness, user ID, SSL inspection",
              ],
            },
          },
          {
            id: "firewall-best-practices",
            title: "Best Practices for Firewall Deployment",
            content: {
              explanation: [
                "Network Segmentation: Divide the network into security zones (Internet, DMZ, Internal, Management). Place firewalls at zone boundaries. Use VLANs and subnets to limit broadcast domains and reduce attack surface.",
                "Rule Management Best Practices: (1) Follow principle of least privilege - deny by default, allow only what's needed, (2) Order rules from most specific to most general, (3) Document all rules with business justification, (4) Regularly review and remove unused rules, (5) Use object groups for manageable rule sets.",
                "High Availability (HA): Deploy firewalls in Active-Passive or Active-Active pairs. Ensures continuity if one device fails. State synchronization keeps both devices aware of connections. Regular failover testing is essential.",
                "Logging and Monitoring: Enable comprehensive logging for all denied traffic and critical allowed traffic. Send logs to SIEM for analysis. Set up alerts for suspicious patterns. Retain logs per compliance requirements.",
                "Regular Maintenance: Keep firmware updated for security patches. Backup configuration regularly. Test rule changes in lab before production. Conduct periodic security audits and penetration tests.",
                "Performance Considerations: Size firewall for expected throughput. Consider SSL inspection overhead. Enable only needed features. Monitor CPU, memory, and connection counts.",
              ],
              keyPoints: [
                "Segment network into security zones",
                "Deny by default, allow only necessary traffic",
                "Document all rules with business justification",
                "Deploy HA pairs for reliability",
                "Centralized logging to SIEM",
                "Regular audits and rule reviews",
                "Keep firmware updated",
              ],
              diagrams: [
                {
                  title: "Network Segmentation with Firewalls",
                  description: "Draw a network diagram: Internet → External Firewall → DMZ (Web servers, Email) → Internal Firewall → Internal Zones (Servers, Users, Management). Label each zone and show firewall placement between zones.",
                  imageUrl: "/diagrams/network_segmentation_diagram.png",
                },
                {
                  title: "Firewall High Availability",
                  description: "Draw Active-Passive HA pair: Two firewall boxes connected via heartbeat link. Show state synchronization between them. Primary handles traffic; Secondary takes over on failure. Show failover arrow.",
                  imageUrl: "/diagrams/firewall_ha.svg",
                },
              ],
              problems: [
                {
                  question: "List five best practices for firewall rule configuration. (5 marks)",
                  answer: "Best Practices for Firewall Rules:\n\n1. PRINCIPLE OF LEAST PRIVILEGE:\n   - Deny all traffic by default\n   - Only allow what is explicitly needed\n\n2. RULE ORDERING:\n   - Place specific rules before general rules\n   - Most frequently matched rules first for performance\n\n3. DOCUMENTATION:\n   - Document each rule with business justification\n   - Include ticket numbers, owner, date created\n\n4. REGULAR REVIEW:\n   - Audit rules quarterly\n   - Remove unused or obsolete rules\n   - Validate rules match current business needs\n\n5. USE OBJECT GROUPS:\n   - Group related IPs, ports, services\n   - Makes rules easier to manage and understand",
                },
                {
                  question: "What is network segmentation? Why is it important for security? (3 marks)",
                  answer: "Network Segmentation:\nDividing a network into smaller, isolated segments or zones using firewalls, VLANs, or subnets.\n\nImportance for Security:\n1. Contains breaches - compromised segment doesn't affect others\n2. Reduces attack surface - attackers can't freely move laterally\n3. Enables granular access control per zone\n4. Supports compliance requirements (PCI-DSS)\n5. Simplifies monitoring and auditing\n\nCommon Zones: Internet, DMZ, Internal, Management, Guest",
                },
              ],
              examTips: [
                "Key principle: Deny by default, allow explicitly",
                "Know HA concepts: Active-Passive, Active-Active, state sync",
                "Logging to SIEM is essential for security monitoring",
                "Regular rule audits prevent rule bloat and security gaps",
              ],
            },
          },
        ],
      },
    ],
  },
  "ethical-hacking": {
    id: "ethical-hacking",
    title: "Ethical Hacking and Network Defense",
    shortTitle: "Ethical Hacking",
    description:
      "Ethical hacking fundamentals, vulnerability analysis, penetration testing, network defense, and IDS deployment.",
    units: [
      {
        id: "unit-1",
        title: "Ethical Hacking Fundamentals & Network Security",
        description: "Core concepts of ethical hacking, penetration testing lifecycle, and network security fundamentals",
        topics: [
          {
            id: "ethical-hacking-fundamentals",
            title: "Fundamentals of Ethical Hacking",
            content: {
              explanation: [
                "Ethical Hacking is the authorized practice of bypassing system security to identify potential data breaches and threats in a network. Ethical hackers use the same tools and techniques as malicious hackers but with permission and for defensive purposes.",
                "Organizations benefit from ethical hacking through: (1) Identifying vulnerabilities before attackers do, (2) Testing security policies and controls, (3) Compliance with security standards (PCI-DSS, HIPAA), (4) Protecting customer data and reputation.",
                "Types of Ethical Hacking Teams: RED TEAM - Offensive security, simulates real attacks to test defenses. BLUE TEAM - Defensive security, detects and responds to attacks. PURPLE TEAM - Collaboration between Red and Blue teams, improves both offense and defense through shared knowledge.",
                "Penetration Testing Lifecycle: (1) Reconnaissance - Gather information about target, (2) Scanning - Identify live hosts, open ports, services, (3) Gaining Access - Exploit vulnerabilities, (4) Maintaining Access - Establish persistence, (5) Covering Tracks - Avoid detection, (6) Reporting - Document findings and recommendations.",
                "Vulnerability Analysis vs Penetration Testing: VA identifies vulnerabilities but doesn't exploit them. PT attempts actual exploitation to prove impact. VA is automated and regular; PT is manual and periodic. Both are essential for comprehensive security assessment.",
              ],
              keyPoints: [
                "Ethical hacking = Authorized security testing",
                "Red Team: Offensive; Blue Team: Defensive; Purple Team: Collaborative",
                "VAPT = Vulnerability Assessment + Penetration Testing",
                "Pentest phases: Recon → Scan → Gain Access → Maintain → Cover → Report",
                "Always get written authorization before testing",
                "Scope defines what can and cannot be tested",
                "Tools: Kali Linux, Metasploit, Burp Suite, Nmap",
              ],
              diagrams: [
                {
                  title: "Penetration Testing Lifecycle",
                  description: "Draw a circular diagram showing 6 phases: Reconnaissance → Scanning → Gaining Access → Maintaining Access → Covering Tracks → Reporting. Show arrows connecting each phase in sequence, with 'Reporting' connecting back to improve 'Reconnaissance' for next cycle.",
                  imageUrl: "/diagrams/pentest_lifecycle_diagram.png",
                },
                {
                  title: "Red vs Blue vs Purple Team",
                  description: "Draw three circles: RED (Attackers - find vulnerabilities, simulate attacks), BLUE (Defenders - detect, respond, protect), PURPLE (Collaboration - shared learning). Show arrows between Red and Blue pointing to Purple, indicating knowledge sharing.",
                  imageUrl: "/diagrams/red_blue_purple_team_diagram.png",
                },
              ],
              problems: [
                {
                  question: "Explain Red, Blue, and Purple teaming in ethical hacking. (5 marks)",
                  answer: "RED TEAM:\n- Offensive security team\n- Simulates real-world attacks\n- Goal: Find vulnerabilities and breach defenses\n- Uses attacker tools and techniques\n- Tests organization's detection capabilities\n\nBLUE TEAM:\n- Defensive security team\n- Monitors, detects, and responds to attacks\n- Goal: Protect assets and detect intrusions\n- Manages SIEM, IDS/IPS, firewalls\n- Performs incident response\n\nPURPLE TEAM:\n- Collaborative approach\n- Combines Red and Blue team efforts\n- Goal: Improve both offense and defense\n- Shares attack insights with defenders\n- Results in better security posture",
                },
                {
                  question: "Describe the lifecycle of penetration testing. (5 marks)",
                  answer: "Penetration Testing Lifecycle:\n\n1. RECONNAISSANCE:\n   - Gather information about target\n   - Passive (OSINT) and Active methods\n\n2. SCANNING:\n   - Identify live hosts, ports, services\n   - Tools: Nmap, Nessus\n\n3. GAINING ACCESS:\n   - Exploit identified vulnerabilities\n   - Use tools like Metasploit\n\n4. MAINTAINING ACCESS:\n   - Establish persistence\n   - Install backdoors if in scope\n\n5. COVERING TRACKS:\n   - Remove evidence of activity\n   - Clear logs (simulate real attacker)\n\n6. REPORTING:\n   - Document all findings\n   - Provide severity ratings\n   - Recommend remediations",
                },
              ],
              examTips: [
                "Remember all 6 phases of penetration testing lifecycle",
                "Know the difference between VA (automated, identifies) and PT (manual, exploits)",
                "Red/Blue/Purple teams - know what each does",
                "Ethical hacking ALWAYS requires written authorization",
              ],
            },
          },
          {
            id: "network-security-fundamentals",
            title: "Network Security & Cryptography",
            content: {
              explanation: [
                "Security Aspects of OSI Model: Each layer has specific security concerns. Layer 2 (Data Link): ARP spoofing, MAC flooding. Layer 3 (Network): IP spoofing, routing attacks. Layer 4 (Transport): TCP SYN floods, session hijacking. Layer 7 (Application): SQL injection, XSS.",
                "Active vs Passive Network Attacks: PASSIVE attacks observe/intercept data without modifying (eavesdropping, traffic analysis) - hard to detect. ACTIVE attacks modify data or disrupt services (DoS, MITM, spoofing) - easier to detect but more damaging.",
                "Cryptography in Network Security: Single Sign-On (SSO) - one authentication for multiple services. Email Encryption: PGP (Pretty Good Privacy) for end-to-end, STARTTLS for transport encryption.",
                "Security Protocols: IPSec - Layer 3 security for VPNs. SSL 3.0 - Deprecated due to POODLE attack. TLS 1.2/1.3 - Current standard for secure communication. HTTPS = HTTP + TLS.",
                "SSL/TLS Attacks: POODLE Attack - Exploits SSL 3.0 fallback, allows decryption of secure connections. DROWN Attack - Exploits SSLv2 to decrypt TLS connections. Both attacks emphasize need to disable old protocols. DNSSEC - Adds authentication to DNS responses, prevents DNS spoofing.",
              ],
              keyPoints: [
                "OSI security: Each layer has specific vulnerabilities",
                "Passive = observe (hard to detect); Active = modify (more damage)",
                "IPSec: Layer 3 VPN security",
                "TLS 1.2/1.3 is current standard; SSL is deprecated",
                "POODLE: SSL 3.0 vulnerability; DROWN: SSLv2 vulnerability",
                "DNSSEC: Authenticates DNS responses",
                "SSO: Single authentication for multiple services",
              ],
              diagrams: [
                {
                  title: "OSI Model Security Threats",
                  description: "Draw OSI 7 layers with security threats at each level: Layer 7 - XSS, SQL injection; Layer 4 - SYN flood, session hijack; Layer 3 - IP spoofing, routing attacks; Layer 2 - ARP spoofing, MAC flood; Layer 1 - Physical access, wiretapping.",
                  imageUrl: "/diagrams/osi_security_diagram.png",
                },
              ],
              problems: [
                {
                  question: "Differentiate between active and passive network attacks with examples. (5 marks)",
                  answer: "PASSIVE ATTACKS:\n- Observe or intercept data without modification\n- Goal: Gain information\n- Hard to detect (no change to data)\n- Examples:\n  1. Eavesdropping/Sniffing\n  2. Traffic analysis\n  3. Network monitoring\n\nACTIVE ATTACKS:\n- Modify data or disrupt services\n- Goal: Alter, disrupt, or gain access\n- Easier to detect (causes changes)\n- Examples:\n  1. Denial of Service (DoS)\n  2. Man-in-the-Middle (MITM)\n  3. IP/MAC Spoofing\n  4. Session Hijacking\n  5. Replay attacks",
                },
                {
                  question: "Explain POODLE and DROWN attacks on SSL/TLS. (5 marks)",
                  answer: "POODLE Attack (Padding Oracle On Downgraded Legacy Encryption):\n- Targets SSL 3.0 protocol\n- Exploits CBC mode padding vulnerability\n- Attacker forces browser to downgrade to SSL 3.0\n- Can decrypt secure cookies byte by byte\n- Mitigation: Disable SSL 3.0 entirely\n\nDROWN Attack (Decrypting RSA with Obsolete and Weakened eNcryption):\n- Targets SSLv2 protocol\n- Can decrypt modern TLS sessions\n- Exploits servers still supporting SSLv2\n- 33% of HTTPS servers were vulnerable\n- Mitigation: Disable SSLv2 on all servers\n\nLesson: Always disable deprecated protocols (SSL 2.0, 3.0).",
                },
              ],
              examTips: [
                "Know security vulnerabilities at each OSI layer",
                "Passive = eavesdropping; Active = modification",
                "Remember: SSL is deprecated; TLS 1.2+ is secure",
                "POODLE = SSL 3.0; DROWN = SSLv2",
              ],
            },
          },
          {
            id: "network-attacks",
            title: "Network Layer Attacks",
            content: {
              explanation: [
                "ARP Cache Poisoning: Attacker sends fake ARP replies to associate their MAC address with a legitimate IP. This allows traffic interception (MITM). Detection: Static ARP entries, ARP monitoring tools. Prevention: Dynamic ARP Inspection (DAI).",
                "MAC Flooding: Attacker floods switch with fake MAC addresses, filling CAM table. Switch then acts like a hub, broadcasting all traffic. Attacker can now sniff all network traffic. Prevention: Port security, limiting MACs per port.",
                "Port Stealing: Attacker sends frames with victim's MAC as source from attacker's port. Switch updates CAM table, directing victim's traffic to attacker. Prevention: Port security, sticky MACs.",
                "DHCP Attacks: DHCP Starvation - Exhaust DHCP pool with fake requests. DHCP Spoofing - Rogue DHCP server provides malicious gateway. Prevention: DHCP snooping, rate limiting.",
                "DNS-based Attacks: DNS Spoofing - Return false DNS responses. DNS Hijacking - Redirect DNS queries. DNS Amplification - DDoS using DNS servers. Prevention: DNSSEC, secure resolvers.",
                "VLAN Hopping: Switch Spoofing - Attacker pretends to be a switch trunk port. Double Tagging - Attacker adds extra VLAN tag to reach other VLANs. Prevention: Disable auto-trunking, native VLAN tagging.",
                "Man-in-the-Middle (MITM): Attacker positions between client and server, intercepting all communication. Can be achieved via ARP spoofing, DNS spoofing, or rogue WiFi. Prevention: Encryption (HTTPS), certificate pinning.",
              ],
              keyPoints: [
                "ARP Poisoning: Fake ARP → associate attacker MAC with legitimate IP",
                "MAC Flooding: Overflow CAM table → switch becomes hub",
                "DHCP attacks: Starvation (exhaust pool) , Spoofing (rogue server)",
                "DNS attacks: Spoofing, Hijacking, Amplification",
                "VLAN Hopping: Switch spoofing or double tagging",
                "MITM: Intercept communication between two parties",
                "Prevention: Port security, DAI, DHCP snooping, DNSSEC",
              ],
              diagrams: [
                {
                  title: "ARP Cache Poisoning Attack",
                  description: "Draw three boxes: Attacker, Victim, Gateway. Show Attacker sending fake ARP 'I am Gateway' to Victim and 'I am Victim' to Gateway. Result: All traffic between Victim and Gateway passes through Attacker.",
                  imageUrl: "/diagrams/arp_poisoning_diagram.png",
                },
                {
                  title: "MITM Attack Flow",
                  description: "Draw: Client → Attacker (intercepts) → Server. Show Attacker can read/modify all traffic. Label: Client thinks it's talking to Server; Server thinks it's talking to Client.",
                  imageUrl: "/diagrams/mitm_attack.svg",
                },
              ],
              problems: [
                {
                  question: "Explain ARP Cache Poisoning attack and its prevention. (5 marks)",
                  answer: "ARP Cache Poisoning Attack:\n\nMethod:\n1. Attacker sends fake ARP replies\n2. Associates attacker's MAC with victim's IP (or gateway's IP)\n3. Victim's ARP cache updated with wrong mapping\n4. Traffic intended for victim/gateway goes to attacker\n5. Enables Man-in-the-Middle attack\n\nImpact:\n- Traffic interception and modification\n- Credential theft\n- Session hijacking\n\nPrevention:\n1. Static ARP entries for critical hosts\n2. Dynamic ARP Inspection (DAI) on switches\n3. ARP monitoring tools (arpwatch)\n4. Network segmentation\n5. Encrypted protocols (HTTPS, SSH)",
                },
                {
                  question: "Describe various DHCP attacks and their countermeasures. (5 marks)",
                  answer: "DHCP Attacks:\n\n1. DHCP STARVATION:\n- Attacker sends many DHCP discovers with spoofed MACs\n- Exhausts IP address pool\n- Legitimate clients can't get IP\n\n2. DHCP SPOOFING:\n- Attacker sets up rogue DHCP server\n- Responds faster than legitimate server\n- Provides malicious gateway/DNS\n- Enables MITM attack\n\nCountermeasures:\n1. DHCP Snooping:\n   - Configure trusted ports (legitimate DHCP server)\n   - Block DHCP responses on untrusted ports\n2. Rate limiting DHCP requests\n3. Port security to limit MACs per port\n4. Network Access Control (NAC)\n5. Monitor for rogue DHCP servers",
                },
              ],
              examTips: [
                "Know attack mechanism AND prevention for each attack",
                "ARP Poisoning enables MITM - remember this connection",
                "MAC Flooding turns switch into hub - enables sniffing",
                "DHCP Snooping prevents DHCP attacks",
              ],
            },
          },
          {
            id: "web-application-security",
            title: "Web Application Security",
            content: {
              explanation: [
                "Web Application Security Threats: Web applications are primary targets due to their accessibility. OWASP Top 10 lists most critical vulnerabilities. Understanding these threats is essential for both offense (testing) and defense (development).",
                "Cross-Site Scripting (XSS): Attacker injects malicious scripts into web pages viewed by other users. Types: Stored XSS (persistent, saved in database), Reflected XSS (non-persistent, in URL parameters), DOM-based XSS (client-side). Impact: Session hijacking, defacement, phishing.",
                "Cross-Site Request Forgery (CSRF): Attacker tricks authenticated user into performing unwanted actions on a web application. Exploits the trust a site has in user's browser. Example: Hidden form that transfers money when victim visits attacker's page. Prevention: CSRF tokens, SameSite cookies.",
                "SQL Injection: Attacker inserts malicious SQL code through user input to manipulate database. Types: In-band (error-based, union-based), Blind (boolean-based, time-based), Out-of-band. Impact: Data theft, data modification, authentication bypass, database takeover. Prevention: Parameterized queries, input validation, WAF.",
                "Remote File Inclusion (RFI): Attacker includes external malicious file in web application. Exploits insecure file include functions (PHP: include, require). Can lead to Remote Code Execution. Prevention: Disable allow_url_include, input validation.",
                "DoS/DDoS Attacks: Denial of Service overwhelms system resources making it unavailable. DDoS uses multiple attacking systems. Types: Volumetric (flood bandwidth), Protocol (exploit protocol weaknesses), Application layer (HTTP floods). Prevention: Rate limiting, CDN, DDoS protection services.",
              ],
              keyPoints: [
                "XSS: Inject scripts into web pages (Stored, Reflected, DOM)",
                "CSRF: Trick user into unwanted actions",
                "SQL Injection: Manipulate database through input",
                "RFI: Include and execute external malicious files",
                "DoS/DDoS: Overwhelm resources to deny service",
                "Prevention: Input validation, CSRF tokens, prepared statements",
                "OWASP Top 10: Reference for web vulnerabilities",
              ],
              diagrams: [
                {
                  title: "SQL Injection Attack",
                  description: "Show: User Input → Application → Database. Normal: 'SELECT * FROM users WHERE id = 1'. Malicious: 'SELECT * FROM users WHERE id = 1 OR 1=1'. Result: Returns all users instead of one. Label: Input validation prevents this.",
                  imageUrl: "/diagrams/sql_injection_diagram.png",
                },
                {
                  title: "XSS Attack Flow",
                  description: "Stored XSS: Attacker submits malicious script → Stored in database → Victim loads page → Script executes in victim's browser → Attacker gets session cookie. Show flow with arrows.",
                  imageUrl: "/diagrams/xss_attack_diagram.png",
                },
              ],
              examples: [
                {
                  title: "SQL Injection Example",
                  problem: "Explain how ' OR 1=1 -- can bypass login authentication.",
                  solution: "Original Query:\nSELECT * FROM users WHERE username='admin' AND password='password'\n\nWith Injection (username: ' OR 1=1 --):\nSELECT * FROM users WHERE username='' OR 1=1 --' AND password='password'\n\nExplanation:\n- ' closes the username string\n- OR 1=1 is always true, matches all rows\n- -- comments out rest of query\n- Result: Returns first user (usually admin)\n- Attacker gains access without password",
                },
              ],
              problems: [
                {
                  question: "Explain XSS attack types and prevention methods. (5 marks)",
                  answer: "Cross-Site Scripting (XSS) Types:\n\n1. STORED XSS:\n- Malicious script saved in database\n- Executes whenever page is loaded\n- Most dangerous type\n- Example: Comment section with script\n\n2. REFLECTED XSS:\n- Script in URL parameter\n- Not stored, reflected back\n- Requires victim to click link\n- Example: Search query in URL\n\n3. DOM-BASED XSS:\n- Occurs in client-side JavaScript\n- No server interaction\n- Modifies DOM environment\n\nPrevention:\n1. Input validation and sanitization\n2. Output encoding (HTML entities)\n3. Content Security Policy (CSP)\n4. HttpOnly flag on cookies\n5. Use security frameworks",
                },
                {
                  question: "What is SQL Injection? Explain with example and prevention. (5 marks)",
                  answer: "SQL Injection:\n\nDefinition: Attacker inserts malicious SQL through user input to manipulate database.\n\nExample:\nLogin form: username='admin' password=[malicious input]\nInput: ' OR '1'='1\nQuery becomes:\nSELECT * FROM users WHERE username='admin' AND password='' OR '1'='1'\nResult: Bypasses authentication (1=1 is always true)\n\nTypes:\n1. In-band: Error-based, Union-based\n2. Blind: Boolean, Time-based\n3. Out-of-band: DNS, HTTP requests\n\nPrevention:\n1. Parameterized queries/Prepared statements\n2. Input validation\n3. Least privilege database accounts\n4. Web Application Firewall\n5. Regular security testing",
                },
              ],
              examTips: [
                "Know all 3 XSS types: Stored, Reflected, DOM-based",
                "SQL Injection example: ' OR 1=1 -- is classic bypass",
                "CSRF uses victim's authenticated session",
                "Prevention methods are as important as attack understanding",
              ],
            },
          },
          {
            id: "ids-firewalls",
            title: "Intrusion Detection Systems & Network Defense",
            content: {
              explanation: [
                "Intrusion Detection System (IDS): Monitors network traffic for suspicious activity and alerts administrators. Types: NIDS (Network-based), HIDS (Host-based). Detection methods: Signature-based (known patterns), Anomaly-based (deviation from baseline).",
                "Snort IDS: Open-source network intrusion detection system. Can perform real-time traffic analysis and packet logging. Uses rules to detect attacks: alert tcp any any -> any 80 (msg:'HTTP Attack'; content:'malicious'; sid:1001;). Modes: Sniffer, Packet Logger, NIDS.",
                "Signature-based Detection: Matches traffic against database of known attack signatures. Pros: Accurate for known attacks, low false positives. Cons: Cannot detect new/unknown attacks (zero-day), requires regular updates.",
                "Anomaly-based Detection: Learns normal behavior baseline, alerts on deviations. Pros: Can detect unknown attacks, adapts to environment. Cons: Higher false positives, requires training period, may miss slow attacks.",
                "Firewalls vs IDS: Firewalls BLOCK traffic based on rules (preventive). IDS DETECTS and alerts on suspicious traffic (detective). IPS combines both - detects AND blocks (preventive). Modern solutions integrate all three.",
                "Honeypots and Honeynets: Decoy systems designed to attract attackers. Honeypot: Single fake system. Honeynet: Network of honeypots. Purpose: Study attacker techniques, early warning, divert from real assets. Must be isolated from production network.",
                "Lab Exercise - Snort Deployment: Install Snort on Linux, configure network interface, write custom signatures to detect specific malicious traffic patterns, generate alerts, analyze logs.",
              ],
              keyPoints: [
                "IDS: Detect and alert; IPS: Detect and block",
                "Snort: Open-source NIDS, uses rules for detection",
                "Signature-based: Known attacks, low false positives",
                "Anomaly-based: Unknown attacks, higher false positives",
                "NIDS: Network-level; HIDS: Host-level",
                "Honeypots: Decoy systems to study attackers",
                "Firewalls: Preventive; IDS: Detective",
              ],
              diagrams: [
                {
                  title: "IDS Deployment Architecture",
                  description: "Draw: Internet → Firewall → IDS (monitoring copy of traffic via SPAN port) → Internal Network. Show IDS connected to SIEM for log analysis and alerting. Label that IDS passively monitors a copy of traffic.",
                  imageUrl: "/diagrams/ids_architecture_diagram.png",
                },
                {
                  title: "Snort Rule Structure",
                  description: "Show rule: alert tcp any any -> any 80 (msg:'Attack'; content:'malicious'; sid:1001;). Break down: Action=alert, Protocol=tcp, Source=any:any, Direction=→, Dest=any:80, Options in parentheses. Explain each component.",
                  imageUrl: "/diagrams/snort_rules.svg",
                },
              ],
              examples: [
                {
                  title: "Custom Snort Rule",
                  problem: "Write a Snort rule to detect potential SQL injection attempts.",
                  solution: "Snort Rule:\nalert tcp any any -> any 80 (msg:\"SQL Injection Attempt\"; flow:to_server,established; content:\"' OR\"; nocase; sid:1000001; rev:1;)\n\nExplanation:\n- alert: Generate alert when matched\n- tcp any any -> any 80: HTTP traffic (port 80)\n- msg: Description of alert\n- flow: Traffic direction to server\n- content: Look for \"' OR\" in payload\n- nocase: Case-insensitive match\n- sid: Unique signature ID\n- rev: Rule revision number",
                },
              ],
              problems: [
                {
                  question: "Compare signature-based and anomaly-based intrusion detection. (5 marks)",
                  answer: "Signature-Based Detection:\n- Matches traffic against known attack patterns\n- Requires signature database\n- Accurate for known attacks\n- Low false positive rate\n- Cannot detect zero-day attacks\n- Needs regular signature updates\n- Example: Snort with rule sets\n\nAnomaly-Based Detection:\n- Learns normal behavior baseline\n- Alerts on deviations from normal\n- Can detect unknown attacks\n- Higher false positive rate\n- Requires training period\n- May miss slow/subtle attacks\n- Example: ML-based IDS\n\nComparison:\n- Signature = Accuracy for known; Anomaly = Coverage for unknown\n- Best approach: Combine both methods",
                },
                {
                  question: "Explain the components of a Snort IDS rule. (5 marks)",
                  answer: "Snort Rule Structure:\nalert tcp $EXTERNAL -> $HOME any 22 (msg:'SSH'; sid:1;)\n\nCOMPONENTS:\n\n1. ACTION: What to do when matched\n   - alert, log, pass, drop, reject\n\n2. PROTOCOL: Network protocol\n   - tcp, udp, icmp, ip\n\n3. SOURCE: Source IP and port\n   - Format: IP port (can use variables)\n\n4. DIRECTION: Traffic direction\n   - -> (source to dest)\n   - <> (bidirectional)\n\n5. DESTINATION: Dest IP and port\n\n6. OPTIONS (in parentheses):\n   - msg: Alert message\n   - content: Pattern to match\n   - sid: Signature ID (unique)\n   - rev: Rule revision\n   - flags: TCP flags\n   - flow: Connection state",
                },
              ],
              examTips: [
                "Know Snort rule syntax: action protocol src -> dst (options)",
                "Signature = known attacks; Anomaly = unknown attacks",
                "IDS = detect/alert; IPS = detect/block",
                "Lab: Be prepared to write simple Snort rules",
                "Honeypots: Decoys to attract and study attackers",
              ],
            },
          },
        ],
      },
    ],
  },
  oomd: {
    id: "oomd",
    title: "Object Oriented Modelling and Design",
    shortTitle: "OOMD",
    description: "Object-oriented concepts, UML modeling, class diagrams, state diagrams, and system/class design.",
    units: [
      {
        id: "unit-1",
        title: "Introduction to Modeling & Class Models",
        description: "Modeling concepts, object-oriented development, and class diagram fundamentals",
        topics: [
          {
            id: "modeling-concepts",
            title: "Modeling Concepts & Object-Oriented Development",
            content: {
              explanation: [
                "Modeling is the process of creating abstract representations of real-world systems to understand, analyze, and design solutions. A model simplifies reality by focusing on relevant aspects while ignoring unnecessary details.",
                "Object-Oriented Development: A software development approach that organizes systems around objects rather than functions. Objects encapsulate data (attributes) and behavior (methods). The approach uses models to represent the structure and behavior of the system throughout the development lifecycle.",
                "Object-Oriented Themes: (1) Abstraction - focus on essential features, hide complexity, (2) Encapsulation - bundle data and methods, (3) Inheritance - create new classes from existing ones, (4) Polymorphism - same interface, different implementations.",
                "Development Process: Requirements Analysis → System Design → Object Design → Implementation. Models are created at each stage: Use case models capture requirements, class diagrams define structure, interaction diagrams show behavior.",
                "Benefits of OO Modeling: Promotes reusability, modularity, and maintainability. Models serve as documentation and communication tools. Early detection of design flaws before coding begins.",
              ],
              keyPoints: [
                "Model = Simplified representation of reality",
                "Object = Data (attributes) + Behavior (methods)",
                "Four OO themes: Abstraction, Encapsulation, Inheritance, Polymorphism",
                "OO Development: Analyze → Design → Implement",
                "Models help communicate and document design decisions",
                "UML is standard notation for OO modeling",
              ],
              diagrams: [
                {
                  title: "Object-Oriented Development Process",
                  description: "Draw a flow: Requirements Analysis (Use Cases) → System Design (Architecture) → Object Design (Class Diagrams) → Implementation (Code). Show models produced at each stage.",
                  imageUrl: "/diagrams/ood_process.svg",
                },
              ],
              problems: [
                {
                  question: "What is modeling? Explain the importance of modeling in software development. (5 marks)",
                  answer: "Modeling:\n\nDefinition: Creating abstract representations of real-world systems to understand, analyze, and design solutions.\n\nImportance in Software Development:\n\n1. UNDERSTANDING:\n- Simplifies complex systems\n- Focuses on relevant aspects\n- Helps stakeholders visualize system\n\n2. COMMUNICATION:\n- Common language between developers\n- Documentation for future reference\n- Bridges gap between technical and business\n\n3. ANALYSIS:\n- Early detection of design flaws\n- Validate requirements before coding\n- Identify missing functionality\n\n4. DESIGN:\n- Blueprint for implementation\n- Enables parallel development\n- Supports code generation tools",
                },
                {
                  question: "Explain the four themes of object orientation. (5 marks)",
                  answer: "Four Themes of Object Orientation:\n\n1. ABSTRACTION:\n- Focus on essential features\n- Hide unnecessary complexity\n- Example: Car abstraction shows drive(), stop() not engine internals\n\n2. ENCAPSULATION:\n- Bundle data and methods together\n- Hide internal implementation\n- Control access through public interface\n- Example: BankAccount hides balance, exposes deposit()\n\n3. INHERITANCE:\n- Create new classes from existing ones\n- Child inherits parent's attributes and methods\n- Promotes code reuse\n- Example: Car extends Vehicle\n\n4. POLYMORPHISM:\n- Same interface, different implementations\n- Object can take many forms\n- Example: draw() works differently for Circle and Rectangle",
                },
              ],
              examTips: [
                "Know all four OO themes with examples",
                "Understand when to use modeling in development",
                "Models = communication + documentation + analysis",
              ],
            },
          },
          {
            id: "class-modeling",
            title: "Class Modeling Fundamentals",
            content: {
              explanation: [
                "Class Diagram is a UML structural diagram that shows the static structure of a system. It depicts classes with their attributes and operations, and relationships between classes.",
                "Class Notation: A class is represented as a rectangle divided into three compartments: (1) Class Name (top), (2) Attributes (middle) - format: visibility name: type, (3) Operations (bottom) - format: visibility name(params): returnType.",
                "Visibility Modifiers: + (public), - (private), # (protected), ~ (package). Control access to attributes and operations.",
                "Object vs Class: Class is a blueprint/template, Object is an instance of a class. Class defines structure, Object holds actual values.",
                "Link and Association: Link is a physical or conceptual connection between objects. Association is a relationship between classes. Association has: Name, Role names, Multiplicity (1, 0..1, *, 1..*), Navigability.",
              ],
              keyPoints: [
                "Class = Name + Attributes + Operations",
                "Visibility: + public, - private, # protected",
                "Class vs Object: Blueprint vs Instance",
                "Association: Relationship between classes",
                "Multiplicity: 1 (exactly one), * (many), 0..1 (optional)",
                "Link: Instance of an association",
              ],
              diagrams: [
                {
                  title: "Class Diagram Notation",
                  description: "Draw a class rectangle with three compartments: Top = 'Student', Middle = '- studentId: int, - name: String', Bottom = '+ enroll(): void, + getGrade(): float'. Show visibility symbols.",
                  imageUrl: "/diagrams/class_diagram_notation.png",
                },
                {
                  title: "Association with Multiplicity",
                  description: "Draw: Student class connected to Course class. Label association 'enrolls'. Show multiplicity: Student 1..* --- * Course. Add role names: student, course.",
                  imageUrl: "/diagrams/association_multiplicity.svg",
                },
              ],
              problems: [
                {
                  question: "Explain the components of a class diagram with notation. (5 marks)",
                  answer: "Class Diagram Components:\n\n1. CLASS:\n- Rectangle with 3 compartments\n- Top: Class name (bold, centered)\n- Middle: Attributes\n- Bottom: Operations\n\n2. ATTRIBUTE NOTATION:\nvisibility name: type = defaultValue\nExample: - balance: float = 0.0\n\n3. OPERATION NOTATION:\nvisibility name(params): returnType\nExample: + deposit(amount: float): void\n\n4. VISIBILITY:\n+ public (accessible everywhere)\n- private (class only)\n# protected (class and subclasses)\n~ package (same package)\n\n5. RELATIONSHIPS:\n- Association (solid line)\n- Aggregation (hollow diamond)\n- Composition (filled diamond)\n- Inheritance (hollow triangle arrow)",
                },
                {
                  question: "Differentiate between class and object with examples. (3 marks)",
                  answer: "CLASS vs OBJECT:\n\nCLASS:\n- Blueprint/Template\n- Defines structure and behavior\n- Does not hold actual data\n- Created at design time\n- Example: 'Student' class with attributes name, id\n\nOBJECT:\n- Instance of a class\n- Holds actual values\n- Exists at runtime\n- Example: student1 with name='John', id=101\n\nAnalogy:\nClass = Cookie cutter (template)\nObject = Actual cookie (instance)",
                },
              ],
              examTips: [
                "Memorize class diagram notation",
                "Practice drawing class diagrams from requirements",
                "Know visibility symbols: +, -, #, ~",
                "Association arrow points to navigable class",
              ],
            },
          },
        ],
      },
      {
        id: "unit-2",
        title: "Advanced Modeling & Behavioral Diagrams",
        description: "Advanced class modeling, state diagrams, and interaction diagrams",
        topics: [
          {
            id: "advanced-class-modeling",
            title: "Advanced Class Modeling",
            content: {
              explanation: [
                "Aggregation and Composition: Both represent whole-part relationships. Aggregation (hollow diamond): Weak relationship, parts can exist independently. Composition (filled diamond): Strong relationship, parts cannot exist without whole.",
                "Generalization (Inheritance): Represents 'is-a' relationship. Child class inherits attributes and operations from parent. Shown with hollow triangle arrow pointing to parent. Example: Car is-a Vehicle.",
                "Abstract Class: Cannot be instantiated, serves as template. Name written in italics. Contains abstract methods (no implementation). Example: Shape is abstract, Circle and Rectangle extend it.",
                "Multiple Inheritance: A class inherits from multiple parents. Supported in some languages (C++), others use interfaces (Java). Creates diamond problem - ambiguity when parents have same method.",
                "Association Class: An association that has its own attributes or operations. Example: Enrollment is an association class between Student and Course, with attributes like grade, enrollmentDate.",
                "Constraints: Additional rules on the model shown in curly braces. Example: {ordered}, {unique}, {subset}. OCL (Object Constraint Language) for complex constraints.",
              ],
              keyPoints: [
                "Aggregation: Hollow diamond, parts can exist independently",
                "Composition: Filled diamond, parts destroyed with whole",
                "Generalization: Is-a, hollow triangle to parent",
                "Abstract class: Italicized, cannot instantiate",
                "Association class: Association with its own attributes",
                "Constraints: Additional rules in {curly braces}",
              ],
              diagrams: [
                {
                  title: "Aggregation vs Composition",
                  description: "Draw: University ◇---- Department (aggregation) and Department ◆---- Professor (composition). Label: Departments exist if University closes, Professors (employees) don't exist without Department.",
                  imageUrl: "/diagrams/aggregation_composition_diagram.png",
                },
                {
                  title: "Generalization Hierarchy",
                  description: "Draw: Vehicle (parent) with hollow triangle arrows from Car and Motorcycle. Show inherited attributes (speed, color) in parent, specific attributes (numDoors, hasSidecar) in children.",
                  imageUrl: "/diagrams/uml_generalization.svg",
                },
              ],
              problems: [
                {
                  question: "Differentiate between aggregation and composition with examples. (5 marks)",
                  answer: "Aggregation vs Composition:\n\nAGGREGATION (Hollow Diamond ◇):\n- Weak 'has-a' relationship\n- Parts CAN exist independently\n- Shared ownership possible\n- Example: Library has Books\n  - Books exist even if Library closes\n- Example: Car has Engine\n  - Engine can be moved to another Car\n\nCOMPOSITION (Filled Diamond ◆):\n- Strong 'has-a' relationship\n- Parts CANNOT exist independently\n- Exclusive ownership\n- Example: House has Rooms\n  - Rooms destroyed when House demolished\n- Example: Order has LineItems\n  - LineItems meaningless without Order\n\nKey Difference: Lifecycle dependency - in composition, part's lifecycle depends on whole.",
                },
                {
                  question: "What is an abstract class? How is it represented in UML? (3 marks)",
                  answer: "Abstract Class:\n\nDefinition:\n- A class that cannot be instantiated directly\n- Serves as template/blueprint for subclasses\n- May contain abstract methods (no implementation)\n- Subclasses must implement abstract methods\n\nUML Representation:\n- Class name in ITALICS\n- Or use stereotype <<abstract>>\n- Abstract methods also in italics\n\nExample:\n┌──────────────────┐\n│     Shape        │  (italicized)\n├──────────────────┤\n│ # color: Color   │\n├──────────────────┤\n│ + draw(): void   │  (italicized = abstract)\n│ + getArea(): float│\n└──────────────────┘",
                },
              ],
              examTips: [
                "Remember: Hollow diamond = Aggregation, Filled = Composition",
                "Abstract = italics, cannot instantiate",
                "Association class connects to association line",
              ],
            },
          },
          {
            id: "state-diagrams",
            title: "State Diagrams",
            content: {
              explanation: [
                "State Diagram (State Machine Diagram): A UML behavioral diagram that shows the states an object can be in and the transitions between states based on events. Used for objects with complex lifecycle.",
                "State: A condition or situation during the life of an object. Shown as rounded rectangle. Has: State name, Entry/Exit actions, Internal activities.",
                "Transition: Change from one state to another. Shown as arrow with label: event [guard] / action. Event triggers transition, guard is condition, action is performed during transition.",
                "Initial and Final States: Initial state shown as filled circle (starting point). Final state shown as bull's eye (circle with dot inside). Every state diagram should have initial state.",
                "Composite States: States that contain sub-states (nested states). Used to manage complexity. Can have parallel regions for concurrent states.",
                "Example - Order State Diagram: States: Pending → Confirmed → Shipped → Delivered. Transitions: place order, confirm payment, ship, deliver. Shows object lifecycle.",
              ],
              keyPoints: [
                "State = Condition of object at a point in time",
                "Transition = State change triggered by event",
                "Format: event [guard] / action",
                "Initial state: Filled circle",
                "Final state: Bull's eye (circle with dot)",
                "Composite state: Contains sub-states",
              ],
              diagrams: [
                {
                  title: "State Diagram for Order",
                  description: "Draw: ● (initial) → Pending --[payment received]-→ Confirmed --[shipped]-→ Shipped --[delivered]-→ Delivered → ⊙ (final). Show events on transitions.",
                  imageUrl: "/diagrams/state_diagram_order.svg",
                },
                {
                  title: "State Notation",
                  description: "Draw rounded rectangle: State name on top, separator line, entry/do/exit actions below. Example: 'Processing' state with 'entry/startTimer', 'do/processData', 'exit/stopTimer'.",
                  imageUrl: "/diagrams/state_notation.svg",
                },
              ],
              problems: [
                {
                  question: "Draw and explain a state diagram for an ATM transaction. (5 marks)",
                  answer: "ATM Transaction State Diagram:\n\nSTATES:\n1. Idle - ATM waiting for card\n2. Card Inserted - Reading card\n3. PIN Entry - Waiting for PIN\n4. Validating - Checking PIN\n5. Menu - Displaying options\n6. Processing - Executing transaction\n7. Complete - Transaction done\n\nTRANSITIONS:\n● → Idle (initial)\nIdle --[insert card]-→ Card Inserted\nCard Inserted --[card read]-→ PIN Entry\nPIN Entry --[PIN entered]-→ Validating\nValidating --[PIN valid]-→ Menu\nValidating --[PIN invalid]-→ PIN Entry\nMenu --[select option]-→ Processing\nProcessing --[success]-→ Complete\nComplete --[eject card]-→ Idle\n\nGuards: [attempts < 3], [valid PIN]",
                },
                {
                  question: "Explain the components of a state transition. (3 marks)",
                  answer: "State Transition Components:\n\nFormat: event [guard] / action\n\n1. EVENT:\n- Trigger that causes transition\n- Can be: Signal, Call, Time, Change\n- Example: buttonPressed, timeout\n\n2. GUARD (optional):\n- Boolean condition in square brackets\n- Transition only if guard is true\n- Example: [balance > 0], [isValid]\n\n3. ACTION (optional):\n- Operation performed during transition\n- Shown after slash /\n- Example: /displayMessage(), /deductAmount()\n\nComplete Example:\nwithdraw [balance >= amount] / updateBalance()",
                },
              ],
              examTips: [
                "Practice drawing state diagrams for common scenarios",
                "Remember transition format: event [guard] / action",
                "States = nouns, Events = verbs",
                "Always include initial state",
              ],
            },
          },
          {
            id: "interaction-diagrams",
            title: "Interaction Diagrams",
            content: {
              explanation: [
                "Interaction Diagrams: UML diagrams that show how objects interact with each other. Two main types: Sequence Diagrams and Collaboration (Communication) Diagrams.",
                "Sequence Diagram: Shows object interactions arranged in time sequence. Objects shown as boxes at top with vertical lifelines. Messages shown as horizontal arrows between lifelines. Time flows downward.",
                "Sequence Diagram Elements: Lifeline (dashed vertical line), Activation bar (thin rectangle showing object is active), Messages (synchronous→, asynchronous→, return--→), Combined fragments (alt, loop, opt).",
                "Collaboration Diagram: Shows same information as sequence diagram but focuses on object relationships rather than time. Objects shown with links, messages numbered to show order.",
                "Use Cases and Scenarios: Sequence diagrams often realize use cases. Each scenario (path through use case) becomes a sequence diagram. Example: 'Withdraw Cash' use case → Sequence diagram with Customer, ATM, Bank objects.",
                "Choosing between diagrams: Sequence = emphasis on time order. Collaboration = emphasis on object relationships. Both show same information, different perspectives.",
              ],
              keyPoints: [
                "Sequence: Objects across top, time flows down",
                "Lifeline: Dashed vertical line from object",
                "Activation: Rectangle showing object is active",
                "Messages: → synchronous, --> asynchronous, <-- return",
                "Collaboration: Objects linked, numbered messages",
                "Combined fragments: alt (if-else), loop, opt (optional)",
              ],
              diagrams: [
                {
                  title: "Sequence Diagram Structure",
                  description: "Draw: Three objects (User, ATM, Bank) as boxes at top. Dashed lifelines below. Show messages: 1. insertCard, 2. readCard, 3. validateCard, 4. cardValid. Activation bars on receiving object.",
                  imageUrl: "/diagrams/sequence_structure.svg",
                },
                {
                  title: "Collaboration Diagram",
                  description: "Draw: Same three objects (User, ATM, Bank) as boxes. Connect with lines. Number messages: 1: insertCard, 2: readCard, 3: validateCard, 4: cardValid. Show message direction with arrows.",
                  imageUrl: "/diagrams/collaboration_diagram.svg",
                },
              ],
              problems: [
                {
                  question: "Explain sequence diagram components with an example. (5 marks)",
                  answer: "Sequence Diagram Components:\n\n1. OBJECTS/PARTICIPANTS:\n- Boxes at top with object name\n- Format: objectName: ClassName\n\n2. LIFELINE:\n- Dashed vertical line from object\n- Represents object's existence over time\n\n3. ACTIVATION BAR:\n- Thin rectangle on lifeline\n- Shows when object is active/executing\n\n4. MESSAGES:\n- Solid arrow → synchronous call\n- Dashed arrow --> asynchronous\n- Dashed arrow <-- return\n- Labeled with method name\n\n5. COMBINED FRAGMENTS:\n- alt: if-else conditions\n- loop: repetition\n- opt: optional execution\n\nEXAMPLE (Login):\nUser → LoginPage: enterCredentials()\nLoginPage → AuthService: validate()\nAuthService → Database: checkUser()\nDatabase --> AuthService: userDetails\nAuthService --> LoginPage: success/failure",
                },
                {
                  question: "Differentiate between sequence and collaboration diagrams. (5 marks)",
                  answer: "Sequence vs Collaboration Diagrams:\n\nSEQUENCE DIAGRAM:\n- Time-ordered interaction\n- Objects at top, time flows down\n- Shows temporal relationships\n- Easy to see order of messages\n- Good for complex scenarios\n- Lifelines show object existence\n\nCOLLABORATION DIAGRAM:\n- Object-relationship focused\n- Objects anywhere, linked together\n- Messages numbered for order\n- Shows structural relationships\n- Good for simple scenarios\n- Compact representation\n\nSIMILARITIES:\n- Both show object interactions\n- Both derived from same model\n- Both realize use cases\n- Can convert between them\n\nWHEN TO USE:\n- Sequence: Complex scenarios, time-critical\n- Collaboration: Simple scenarios, relationships",
                },
              ],
              examTips: [
                "Practice drawing sequence diagrams for use cases",
                "Remember message types: sync, async, return",
                "Numbering in collaboration shows message order",
                "Combined fragments: alt=if/else, loop=while, opt=if",
              ],
            },
          },
        ],
      },
      {
        id: "unit-3",
        title: "System Design & Class Design",
        description: "System architecture, class design principles, and design optimization",
        topics: [
          {
            id: "system-design",
            title: "System Design Overview",
            content: {
              explanation: [
                "System Design: The process of defining the architecture, components, modules, interfaces, and data for a system to satisfy specified requirements. Transforms analysis model into design model.",
                "System Architecture: High-level structure of the system. Includes: Subsystem decomposition, Hardware/software allocation, Concurrency, Data management strategy, Security considerations.",
                "Subsystem Decomposition: Breaking system into manageable parts. Criteria: High cohesion within subsystem, Low coupling between subsystems. Layered architecture (Presentation, Business Logic, Data) is common pattern.",
                "Architecture Patterns: Client-Server (centralized server), Peer-to-Peer (distributed), MVC (Model-View-Controller), Microservices (independent services), Layered (separation of concerns).",
                "Design Criteria: Correctness (meets requirements), Efficiency (performance), Maintainability (easy to modify), Reliability (handles failures), Reusability (components can be reused), Security (protected from threats).",
                "Trade-offs: Often need to balance competing criteria. Example: Efficiency vs Maintainability (optimized code harder to read), Security vs Usability (more security = more friction).",
              ],
              keyPoints: [
                "System design = Architecture + Components + Interfaces",
                "Subsystem decomposition: High cohesion, Low coupling",
                "Common patterns: MVC, Layered, Client-Server, Microservices",
                "Design criteria: Correctness, Efficiency, Maintainability",
                "Trade-offs between competing quality attributes",
                "Architecture decisions are hard to change later",
              ],
              diagrams: [
                {
                  title: "Layered Architecture",
                  description: "Draw three horizontal layers: Top = Presentation Layer (UI), Middle = Business Logic Layer (Processing), Bottom = Data Access Layer (Database). Show arrows for allowed communication between adjacent layers.",
                  imageUrl: "/diagrams/layered_architecture.svg",
                },
                {
                  title: "MVC Pattern",
                  description: "Draw three boxes: Model (data), View (UI), Controller (logic). Show: View observes Model, Controller updates Model, Controller selects View, User interacts with View. Label responsibilities.",
                  imageUrl: "/diagrams/mvc_pattern.svg",
                },
              ],
              problems: [
                {
                  question: "Explain the criteria for subsystem decomposition. (5 marks)",
                  answer: "Subsystem Decomposition Criteria:\n\n1. HIGH COHESION:\n- Elements within subsystem closely related\n- Single responsibility\n- Easy to understand and maintain\n- Elements should belong together\n\n2. LOW COUPLING:\n- Minimal dependencies between subsystems\n- Changes in one don't affect others\n- Clear, narrow interfaces\n- Promotes independent development\n\n3. LAYERS OF ABSTRACTION:\n- Higher layers use lower layers\n- Lower layers don't know about higher\n- Example: UI → Business → Data\n\n4. PARTITIONS:\n- Vertical slices of functionality\n- Independent feature areas\n- Example: User, Product, Order subsystems\n\n5. CLOSED vs OPEN ARCHITECTURE:\n- Closed: Only adjacent layer access\n- Open: Any layer can access any other",
                },
                {
                  question: "Explain MVC architecture pattern with example. (5 marks)",
                  answer: "MVC (Model-View-Controller) Pattern:\n\n1. MODEL:\n- Contains data and business logic\n- Independent of UI\n- Notifies View of changes\n- Example: User object with name, email\n\n2. VIEW:\n- Displays data to user\n- Observes Model for changes\n- Handles presentation logic\n- Example: HTML page, mobile screen\n\n3. CONTROLLER:\n- Handles user input\n- Updates Model based on actions\n- Selects appropriate View\n- Example: LoginController processes form\n\nFLOW:\n1. User interacts with View\n2. View sends input to Controller\n3. Controller processes, updates Model\n4. Model notifies View of changes\n5. View refreshes with new data\n\nBENEFITS:\n- Separation of concerns\n- Parallel development\n- Easier testing\n- Multiple views for same model",
                },
              ],
              examTips: [
                "Know common architecture patterns (MVC, Layered)",
                "High cohesion + Low coupling = Good design",
                "System design decisions are hard to change",
                "Trade-offs: Balance competing requirements",
              ],
            },
          },
          {
            id: "class-design",
            title: "Class Design & Design Optimization",
            content: {
              explanation: [
                "Class Design: Elaborating the analysis model classes into implementation-ready design classes. Adds detail: Operations with signatures, Visibility, Data types, Algorithms.",
                "Design Principles (SOLID): S - Single Responsibility (one reason to change), O - Open/Closed (open for extension, closed for modification), L - Liskov Substitution (subtypes must be substitutable), I - Interface Segregation (many specific interfaces), D - Dependency Inversion (depend on abstractions).",
                "Designing Operations: Specify complete method signatures. Consider: Parameters, Return types, Visibility, Exceptions. Use descriptive names following conventions (getCUstomerName, calculateTotal).",
                "Designing Associations: Implement using references/pointers. One-to-many: Collection on 'many' side. Many-to-many: May need association class. Add navigation direction for efficiency.",
                "Design Optimization: Improve design for specific quality attributes. Access path optimization (add redundant associations for frequent queries). Indices and caching for database access. Lazy loading for memory optimization.",
                "Refactoring: Restructure code without changing behavior. Common refactorings: Extract Method, Extract Class, Move Method, Rename. Improve design incrementally.",
              ],
              keyPoints: [
                "Class design adds implementation details to analysis classes",
                "SOLID principles guide good design",
                "Single Responsibility: One class, one job",
                "Open/Closed: Extend without modifying",
                "Optimize for specific quality attributes",
                "Refactoring: Improve design without changing behavior",
              ],
              diagrams: [
                {
                  title: "SOLID Principles",
                  description: "Draw five boxes stacked vertically: S-Single Responsibility, O-Open/Closed, L-Liskov Substitution, I-Interface Segregation, D-Dependency Inversion. Add one-line description for each.",
                  imageUrl: "/diagrams/solid_principles.svg",
                },
              ],
              problems: [
                {
                  question: "Explain the SOLID principles with examples. (10 marks)",
                  answer: "SOLID Principles:\n\n1. SINGLE RESPONSIBILITY (S):\n- Class should have one reason to change\n- Bad: UserManager handles auth AND reports\n- Good: Separate AuthService and ReportService\n\n2. OPEN/CLOSED (O):\n- Open for extension, closed for modification\n- Bad: Modify switch for new shapes\n- Good: Shape interface, add new classes\n\n3. LISKOV SUBSTITUTION (L):\n- Subtypes must be substitutable for base\n- Bad: Square extends Rectangle, breaks setWidth\n- Good: Both implement Shape interface\n\n4. INTERFACE SEGREGATION (I):\n- Many specific interfaces better than one general\n- Bad: IWorker with work() and eat()\n- Good: IWorkable and IFeedable separate\n\n5. DEPENDENCY INVERSION (D):\n- Depend on abstractions, not concretions\n- Bad: HighLevel creates LowLevel\n- Good: Both depend on interface\n\nBENEFITS:\n- Maintainable, flexible, testable code",
                },
                {
                  question: "What is refactoring? Give common refactoring techniques. (5 marks)",
                  answer: "Refactoring:\n\nDefinition: Restructuring existing code without changing its external behavior to improve design, readability, and maintainability.\n\nCOMMON TECHNIQUES:\n\n1. EXTRACT METHOD:\n- Long method → smaller methods\n- Improves readability and reuse\n\n2. EXTRACT CLASS:\n- Class doing too much → split into two\n- Single responsibility\n\n3. MOVE METHOD:\n- Move method to class that uses it most\n- Feature envy elimination\n\n4. RENAME:\n- Better names for variables, methods, classes\n- Self-documenting code\n\n5. REPLACE CONDITIONAL:\n- Complex if-else → polymorphism\n- Strategy pattern often helps\n\n6. INLINE METHOD:\n- Method body as simple as name\n- Remove unnecessary indirection\n\nWHEN TO REFACTOR:\n- Code smells detected\n- Before adding new features\n- During code review",
                },
              ],
              examTips: [
                "Memorize SOLID with one example each",
                "Refactoring changes structure, not behavior",
                "Class design = Analysis + Implementation details",
                "Common question: Apply SOLID to given scenario",
              ],
            },
          },
        ],
      },
    ],
  },
  "life-skills": {
    id: "life-skills",
    title: "Life Skills for Engineers",
    shortTitle: "Life Skills",
    description:
      "Essential soft skills including time management, health, stress management, effective habits, and collaborative learning.",
    units: [
      {
        id: "unit-1",
        title: "Time & Health Management",
        description: "Planning essentials, time management techniques, and physical/mental health",
        topics: [
          {
            id: "time-management",
            title: "Time Management",
            content: {
              explanation: [
                "Time Management is the process of planning and controlling how much time to spend on specific activities. Good time management enables you to work smarter, not harder, achieving more in less time even under pressure.",
                "Planning Essentials: (1) Set clear goals - know what you want to achieve, (2) Prioritize tasks - not everything is equally important, (3) Create schedules - allocate time blocks, (4) Review and adjust - adapt to changing circumstances.",
                "Eisenhower Matrix (Urgent-Important Matrix): Quadrant 1 (Urgent + Important): Do first - crises, deadlines. Quadrant 2 (Not Urgent + Important): Schedule - planning, prevention, development. Quadrant 3 (Urgent + Not Important): Delegate - interruptions, some meetings. Quadrant 4 (Not Urgent + Not Important): Eliminate - time wasters.",
                "Time Management Techniques: (1) Pomodoro Technique - 25 min work + 5 min break, (2) Time Blocking - schedule fixed times for tasks, (3) Eat the Frog - do hardest task first, (4) 2-Minute Rule - if it takes less than 2 min, do it now.",
                "Common Time Wasters: Procrastination, excessive social media, unclear priorities, poor planning, inability to say no, multitasking (context switching). Awareness is the first step to elimination.",
              ],
              keyPoints: [
                "Set clear, prioritized goals",
                "Eisenhower Matrix: Urgent/Important classification",
                "Quadrant 2 (Important, Not Urgent) = Prevention and growth",
                "Pomodoro: 25 min focus + 5 min break",
                "Eat the Frog: Tackle hardest task first",
                "Avoid multitasking - context switching wastes time",
              ],
              diagrams: [
                {
                  title: "Eisenhower Matrix",
                  description: "Draw 2x2 grid. Top-left: Q1 (Urgent + Important) = DO. Top-right: Q2 (Not Urgent + Important) = SCHEDULE. Bottom-left: Q3 (Urgent + Not Important) = DELEGATE. Bottom-right: Q4 (Neither) = ELIMINATE.",
                  imageUrl: "/diagrams/eisenhower_matrix.svg",
                },
              ],
              problems: [
                {
                  question: "Explain the Eisenhower Matrix with examples for each quadrant. (5 marks)",
                  answer: "Eisenhower Matrix:\n\nQUADRANT 1 - DO FIRST (Urgent + Important):\n- Crisis situations\n- Pressing deadlines\n- Medical emergencies\nAction: Handle immediately\n\nQUADRANT 2 - SCHEDULE (Not Urgent + Important):\n- Long-term planning\n- Skill development\n- Exercise, health\n- Relationship building\nAction: Schedule dedicated time\n\nQUADRANT 3 - DELEGATE (Urgent + Not Important):\n- Some emails/calls\n- Certain meetings\n- Others' minor issues\nAction: Delegate if possible\n\nQUADRANT 4 - ELIMINATE (Neither):\n- Excessive social media\n- Time-wasting activities\n- Trivial busy work\nAction: Minimize or eliminate\n\nKey Insight: Focus on Quadrant 2 to reduce Quadrant 1 crises.",
                },
                {
                  question: "Describe three time management techniques. (5 marks)",
                  answer: "Time Management Techniques:\n\n1. POMODORO TECHNIQUE:\n- Work in 25-minute focused intervals\n- Take 5-minute breaks between\n- After 4 pomodoros, take 15-30 min break\n- Helps maintain focus and prevent burnout\n\n2. TIME BLOCKING:\n- Divide day into time blocks\n- Assign specific tasks to each block\n- Treats time like an appointment\n- Reduces decision fatigue\n\n3. EAT THE FROG:\n- Identify most challenging task\n- Complete it first thing in morning\n- Uses peak energy on important work\n- Rest of day feels easier\n\nBonus: 2-MINUTE RULE:\n- If task takes < 2 minutes, do immediately\n- Prevents small tasks from piling up",
                },
              ],
              examTips: [
                "Memorize Eisenhower Matrix quadrants",
                "Know 3-4 time management techniques with explanations",
                "Quadrant 2 is key to proactive time management",
                "Common question: Apply matrix to given scenarios",
              ],
            },
          },
          {
            id: "health-management",
            title: "Physical and Mental Health",
            content: {
              explanation: [
                "Physical Health for Engineers: Sedentary work leads to health issues. Key practices: Regular exercise (30 min/day), Ergonomic workspace setup, Regular breaks (20-20-20 rule: every 20 min, look 20 feet away for 20 seconds), Adequate sleep (7-8 hours), Healthy nutrition.",
                "Mental Health Awareness: High-stress profession with burnout risk. Signs of burnout: Exhaustion, cynicism, reduced productivity. Prevention: Work-life boundaries, regular breaks, social connections, professional help when needed.",
                "Ergonomics: The science of designing workplaces to fit users. Monitor at eye level, Chair supports lower back, Feet flat on floor, Wrists neutral when typing. Prevents repetitive strain injuries (RSI), back pain, eye strain.",
                "Sleep and Productivity: Sleep deprivation reduces cognitive function, creativity, and problem-solving ability. Quality sleep improves memory consolidation, decision-making, and emotional regulation. Avoid screens before bed, maintain consistent sleep schedule.",
                "Exercise and Brain Function: Physical activity increases blood flow to brain, releases endorphins, reduces stress hormones. Even short walks improve focus. Regular exercise linked to better memory and learning.",
              ],
              keyPoints: [
                "Sedentary work requires conscious movement",
                "20-20-20 rule: Every 20 min, look 20 ft away for 20 sec",
                "Ergonomics: Monitor, chair, posture alignment",
                "Sleep: 7-8 hours, affects cognition and mood",
                "Exercise: Improves brain function and reduces stress",
                "Burnout signs: Exhaustion, cynicism, low productivity",
              ],
              diagrams: [
                {
                  title: "Ergonomic Workstation",
                  description: "Draw side view of person at desk: Monitor at eye level (arm's length away), Chair with lumbar support, Elbows at 90°, Feet flat on floor. Label each ergonomic adjustment.",
                },
              ],
              problems: [
                {
                  question: "Explain the importance of physical health for engineers with recommendations. (5 marks)",
                  answer: "Physical Health for Engineers:\n\nIMPORTANCE:\n1. Sedentary work causes health issues\n2. Physical health affects mental performance\n3. Prevention of chronic conditions\n4. Sustained productivity over career\n\nRECOMMENDATIONS:\n\n1. REGULAR EXERCISE:\n- 30 minutes daily\n- Mix cardio and strength training\n- Even walking helps\n\n2. ERGONOMIC SETUP:\n- Monitor at eye level\n- Proper chair support\n- Keyboard and mouse position\n\n3. REGULAR BREAKS:\n- 20-20-20 rule for eyes\n- Stand and stretch hourly\n- Short walks\n\n4. NUTRITION:\n- Balanced diet\n- Limit caffeine\n- Stay hydrated\n\n5. SLEEP:\n- 7-8 hours quality sleep\n- Consistent schedule\n- Limit screens before bed",
                },
                {
                  question: "What is burnout? Describe its signs and prevention. (5 marks)",
                  answer: "Burnout:\n\nDEFINITION:\nState of chronic workplace stress that has not been successfully managed, leading to energy depletion, mental distance from job, and reduced effectiveness.\n\nSIGNS:\n1. Physical exhaustion - constant fatigue\n2. Emotional exhaustion - feeling drained\n3. Cynicism - negative attitude toward work\n4. Reduced productivity - decreased performance\n5. Detachment - withdrawal from colleagues\n6. Health issues - headaches, insomnia\n\nPREVENTION:\n1. Set work-life boundaries\n2. Take regular breaks and vacations\n3. Learn to say no to overcommitment\n4. Maintain social connections\n5. Exercise and sleep well\n6. Seek help when needed\n7. Find meaning in work\n8. Practice stress management",
                },
              ],
              examTips: [
                "Know 20-20-20 rule for eye strain prevention",
                "Understand link between physical health and productivity",
                "Burnout: Know signs and prevention strategies",
                "Ergonomics: Common exam topic",
              ],
            },
          },
        ],
      },
      {
        id: "unit-2",
        title: "Habits & Stress Management",
        description: "The 7 Habits of Highly Effective People and stress management techniques",
        topics: [
          {
            id: "seven-habits",
            title: "The 7 Habits of Highly Effective People",
            content: {
              explanation: [
                "The 7 Habits (Stephen Covey) is a framework for personal and professional effectiveness. The habits progress from dependence to independence to interdependence.",
                "Private Victory (Independence): Habit 1 - Be Proactive: Take responsibility for your life. Focus on what you can control (Circle of Influence). React based on values, not emotions. Habit 2 - Begin with the End in Mind: Define clear vision and goals. Write personal mission statement. Visualize desired outcomes. Habit 3 - Put First Things First: Prioritize important over urgent. Use Quadrant 2 thinking. Schedule priorities, don't prioritize schedule.",
                "Public Victory (Interdependence): Habit 4 - Think Win-Win: Seek mutually beneficial solutions. Abundance mentality vs scarcity. Build trust in relationships. Habit 5 - Seek First to Understand, Then to Be Understood: Empathic listening before speaking. Understand others' perspectives. Diagnose before prescribing. Habit 6 - Synergize: Value differences, combine strengths. Whole is greater than sum of parts. Creative cooperation produces better solutions.",
                "Renewal: Habit 7 - Sharpen the Saw: Continuous self-renewal in four dimensions: Physical (exercise, nutrition), Mental (learning, reading), Social/Emotional (relationships), Spiritual (meditation, values). Regular renewal prevents burnout.",
                "Maturity Continuum: Dependence ('you take care of me') → Independence ('I can do it') → Interdependence ('we can do it together'). Highest effectiveness comes from interdependence.",
              ],
              keyPoints: [
                "Habits 1-3: Private Victory (Independence)",
                "Habits 4-6: Public Victory (Interdependence)",
                "Habit 7: Renewal (Sharpen the Saw)",
                "Be Proactive: Focus on Circle of Influence",
                "Begin with End in Mind: Personal mission statement",
                "Put First Things First: Quadrant 2 focus",
                "Win-Win: Abundance mentality",
                "Seek First to Understand: Empathic listening",
                "Synergize: Creative cooperation",
              ],
              diagrams: [
                {
                  title: "The 7 Habits Framework",
                  description: "Draw pyramid or staircase: Base: Habits 1-3 (Private Victory/Independence), Middle: Habits 4-6 (Public Victory/Interdependence), Top: Habit 7 (Renewal). Show progression from dependence to interdependence.",
                },
                {
                  title: "Covey's Circle of Influence",
                  description: "Draw two concentric circles: Inner = Circle of Influence (what you CAN control - your actions, responses, choices). Outer = Circle of Concern (what you care about but can't control - weather, others' actions). Label: Focus on inner circle.",
                  imageUrl: "/diagrams/circle_of_influence.svg",
                },
              ],
              problems: [
                {
                  question: "Explain the 7 Habits of Highly Effective People. (10 marks)",
                  answer: "The 7 Habits (Stephen Covey):\n\nPRIVATE VICTORY (Independence):\n\n1. BE PROACTIVE:\n- Take responsibility for your life\n- Focus on Circle of Influence\n- Choose responses based on values\n\n2. BEGIN WITH END IN MIND:\n- Start with clear vision\n- Create personal mission statement\n- Define long-term goals\n\n3. PUT FIRST THINGS FIRST:\n- Prioritize important over urgent\n- Focus on Quadrant 2 activities\n- Schedule your priorities\n\nPUBLIC VICTORY (Interdependence):\n\n4. THINK WIN-WIN:\n- Seek mutually beneficial outcomes\n- Abundance mentality\n- Build trust and relationships\n\n5. SEEK FIRST TO UNDERSTAND:\n- Practice empathic listening\n- Diagnose before prescribing\n- Understand before being understood\n\n6. SYNERGIZE:\n- Value differences\n- Creative cooperation\n- Whole greater than parts\n\nRENEWAL:\n\n7. SHARPEN THE SAW:\n- Physical renewal (exercise)\n- Mental renewal (learning)\n- Social/emotional renewal (relationships)\n- Spiritual renewal (values, meditation)",
                },
                {
                  question: "What is proactive behavior? Explain with Circle of Influence concept. (5 marks)",
                  answer: "Proactive Behavior (Habit 1):\n\nDEFINITION:\nTaking responsibility for your life and actions rather than blaming external factors. Choosing responses based on values rather than reacting emotionally.\n\nPROACTIVE vs REACTIVE:\n- Reactive: 'There's nothing I can do'\n- Proactive: 'Let's look at alternatives'\n- Reactive: 'He makes me so angry'\n- Proactive: 'I control my response'\n\nCIRCLE OF INFLUENCE:\n\nCircle of Concern (Outer):\n- Things you care about but can't control\n- Weather, economy, others' behavior\n\nCircle of Influence (Inner):\n- Things you CAN control or influence\n- Your actions, attitudes, responses\n\nPROACTIVE FOCUS:\n- Focus on Circle of Influence\n- Take action where you have control\n- Circle of Influence expands over time\n- Reactive people focus on concerns → frustration",
                },
              ],
              examTips: [
                "Memorize all 7 habits in order",
                "Know: 1-3 = Private, 4-6 = Public, 7 = Renewal",
                "Circle of Influence is key concept for Habit 1",
                "Quadrant 2 connects Habit 3 to time management",
              ],
            },
          },
          {
            id: "stress-management",
            title: "Stress Management",
            content: {
              explanation: [
                "Stress is the body's response to demands or challenges. Some stress (eustress) is positive and motivating. Excessive stress (distress) is harmful. Engineering is a high-stress field due to deadlines, complexity, and responsibility.",
                "Types of Stress: Acute stress (short-term, immediate response), Episodic acute stress (frequent acute stress), Chronic stress (long-term, ongoing). Chronic stress leads to health issues: heart disease, weakened immunity, mental health problems.",
                "Stress Response (Fight or Flight): When stressed, body releases cortisol and adrenaline. Heart rate increases, muscles tense, breathing quickens. Helpful for immediate danger, harmful if constant.",
                "Stress Management Techniques: (1) Relaxation - deep breathing, meditation, progressive muscle relaxation, (2) Physical activity - releases endorphins, reduces cortisol, (3) Time management - reduces overwhelm, (4) Social support - talk to friends, family, (5) Cognitive reframing - change perspective on stressors.",
                "Mindfulness and Meditation: Practice of being present in the moment without judgment. Reduces anxiety and stress hormones. Even 10 minutes daily shows benefits. Apps and guided meditations available. Develops awareness of thoughts and reactions.",
              ],
              keyPoints: [
                "Eustress (positive) vs Distress (negative)",
                "Chronic stress leads to health issues",
                "Fight-or-flight: Body's stress response",
                "Techniques: Relaxation, exercise, time management",
                "Social support is crucial for stress management",
                "Mindfulness: Present-moment awareness",
                "Deep breathing activates relaxation response",
              ],
              diagrams: [
                {
                  title: "Stress Response Cycle",
                  description: "Draw cycle: Stressor → Perception → Hormones (cortisol, adrenaline) → Physical Response (heart rate, muscle tension) → Behavior. Show intervention points: Change perception, relaxation techniques.",
                },
              ],
              problems: [
                {
                  question: "What is stress? Explain different types of stress. (5 marks)",
                  answer: "Stress:\n\nDEFINITION:\nThe body's physical, mental, and emotional response to demands, challenges, or perceived threats.\n\nTYPES OF STRESS:\n\n1. EUSTRESS (Positive Stress):\n- Motivating and energizing\n- Short-term, manageable\n- Examples: New job excitement, competition\n- Enhances performance\n\n2. DISTRESS (Negative Stress):\n- Overwhelming and harmful\n- Decreases performance\n- Examples: Excessive workload, conflict\n\n3. ACUTE STRESS:\n- Short-term, immediate response\n- Specific event trigger\n- Body returns to normal after\n\n4. EPISODIC ACUTE STRESS:\n- Frequent acute stress episodes\n- 'Always rushing' personality\n- Pattern of stress reactions\n\n5. CHRONIC STRESS:\n- Long-term, ongoing\n- Most damaging type\n- Examples: Difficult job, relationship issues\n- Leads to health problems",
                },
                {
                  question: "Describe five stress management techniques. (5 marks)",
                  answer: "Stress Management Techniques:\n\n1. RELAXATION TECHNIQUES:\n- Deep breathing exercises\n- Progressive muscle relaxation\n- Guided meditation\n- Yoga and stretching\n\n2. PHYSICAL EXERCISE:\n- Releases endorphins\n- Reduces stress hormones\n- Improves sleep\n- Even 20-minute walks help\n\n3. TIME MANAGEMENT:\n- Reduces overwhelm\n- Prioritize tasks\n- Use Eisenhower Matrix\n- Learn to say no\n\n4. SOCIAL SUPPORT:\n- Talk to friends and family\n- Join support groups\n- Professional counseling\n- Don't isolate\n\n5. COGNITIVE REFRAMING:\n- Change perspective on stressors\n- Challenge negative thoughts\n- Focus on what you can control\n- See challenges as opportunities\n\nBonus: MINDFULNESS\n- Present-moment awareness\n- Non-judgmental observation\n- 10 minutes daily helps",
                },
              ],
              examTips: [
                "Know difference: Eustress (positive) vs Distress (negative)",
                "List 5+ stress management techniques",
                "Understand fight-or-flight response",
                "Chronic stress is most harmful type",
              ],
            },
          },
        ],
      },
      {
        id: "unit-3",
        title: "Learning & Collaboration",
        description: "Effective learning methods and collaborative teamwork",
        topics: [
          {
            id: "learning-methods",
            title: "Effective Learning Methods",
            content: {
              explanation: [
                "Learning How to Learn: Understanding how to learn effectively is a meta-skill that improves all other learning. Key concepts: Focused vs Diffused thinking, Spaced repetition, Active recall, Chunking.",
                "Focused vs Diffused Mode: Focused mode - concentrated attention on specific task, uses prefrontal cortex. Diffused mode - relaxed, mind-wandering state, makes connections. Both modes necessary; alternate between them. Take breaks to allow diffused thinking.",
                "Spaced Repetition: Reviewing material at increasing intervals. Example: Review after 1 day, then 3 days, then 1 week. More effective than cramming. Uses forgetting curve to optimal advantage. Tools: Anki, flashcard apps.",
                "Active Recall: Actively retrieving information rather than passive review. Test yourself instead of re-reading. Creates stronger memory pathways. More effective than highlighting or re-reading. Use practice problems and questions.",
                "Chunking: Breaking complex information into smaller, manageable chunks. Group related concepts together. Build understanding from smaller pieces. Example: Learn programming syntax → then patterns → then architecture.",
                "Feynman Technique: Learn by explaining simply. Steps: (1) Study concept, (2) Explain as if teaching a child, (3) Identify gaps in understanding, (4) Review and simplify. If you can't explain simply, you don't understand it well enough.",
              ],
              keyPoints: [
                "Focused mode: Concentrated attention",
                "Diffused mode: Relaxed, makes connections",
                "Spaced repetition: Review at increasing intervals",
                "Active recall: Test yourself, don't just re-read",
                "Chunking: Break into manageable pieces",
                "Feynman Technique: Explain simply to learn deeply",
                "Take breaks to allow diffused mode thinking",
              ],
              diagrams: [
                {
                  title: "The Forgetting Curve & Spaced Repetition",
                  description: "Draw graph: X-axis = Time, Y-axis = Retention. Show steep forgetting curve declining over time. Then show how repeated reviews at increasing intervals (1d, 3d, 1w, 1m) keep retention high.",
                },
              ],
              problems: [
                {
                  question: "Explain focused and diffused modes of thinking. (5 marks)",
                  answer: "Focused vs Diffused Thinking:\n\nFOCUSED MODE:\n- Concentrated, intense attention\n- Uses prefrontal cortex\n- Good for familiar problems\n- Step-by-step analysis\n- Active when studying/coding\n- Limited perspective\n\nDIFFUSED MODE:\n- Relaxed, resting state\n- Broad, big-picture thinking\n- Makes unexpected connections\n- Creativity and insights\n- Active during breaks, walks, sleep\n- 'Eureka' moments happen here\n\nHOW TO USE BOTH:\n1. Work in focused mode on problem\n2. Take a break if stuck\n3. Let diffused mode process\n4. Return to focused mode\n5. Insight often emerges\n\nEXAMPLE:\nStuck on a bug → take a walk → solution comes to mind → return and implement. Both modes are essential for learning and problem-solving.",
                },
                {
                  question: "What is the Feynman Technique? How does it improve learning? (5 marks)",
                  answer: "Feynman Technique:\n\nDEFINITION:\nA learning method that uses simple explanation to deepen understanding, named after physicist Richard Feynman.\n\nSTEPS:\n\n1. STUDY THE CONCEPT:\n- Read and understand the topic\n- Take initial notes\n\n2. EXPLAIN SIMPLY:\n- Pretend to teach a child\n- Use simple words, no jargon\n- Draw diagrams if helpful\n\n3. IDENTIFY GAPS:\n- Where did explanation break down?\n- What couldn't you simplify?\n- These are knowledge gaps\n\n4. REVIEW AND SIMPLIFY:\n- Go back to source material\n- Fill gaps in understanding\n- Simplify explanation further\n\nWHY IT WORKS:\n- Exposes gaps in understanding\n- Forces clarity of thought\n- Creates deeper connections\n- Simple explanation = true understanding\n- Active rather than passive learning",
                },
              ],
              examTips: [
                "Know focused vs diffused mode differences",
                "Spaced repetition: Review at increasing intervals",
                "Active recall beats passive re-reading",
                "Feynman Technique: Explain simply to learn",
              ],
            },
          },
          {
            id: "collaboration",
            title: "Collaboration and Teamwork",
            content: {
              explanation: [
                "Collaboration is working together toward shared goals, combining diverse skills and perspectives. In engineering, complex problems require teamwork. Collaboration skills are as important as technical skills.",
                "Tuckman's Team Development Stages: (1) Forming - Team meets, polite, roles unclear, (2) Storming - Conflicts arise, personalities clash, (3) Norming - Rules established, cooperation develops, (4) Performing - Team works efficiently toward goals, (5) Adjourning - Project ends, team disbands.",
                "Effective Communication: Clear, concise, and respectful. Active listening - fully attend, ask clarifying questions. Written communication - document decisions, share knowledge. Feedback - give constructive, receive openly.",
                "Conflict Resolution: Conflicts are normal and can be productive. Approaches: Avoidance (temporary), Accommodation (give in), Competition (win-lose), Compromise (meet halfway), Collaboration (win-win). Best approach depends on situation; collaboration often best for team issues.",
                "Psychological Safety: Team members feel safe to take risks and be vulnerable. Can admit mistakes without punishment. Speak up with ideas and concerns. Key factor in high-performing teams (Google research).",
                "Diversity and Inclusion: Diverse teams produce better solutions. Different perspectives prevent groupthink. Inclusive environment where all voices heard. Respect and value differences in background, experience, thinking styles.",
              ],
              keyPoints: [
                "Tuckman: Forming → Storming → Norming → Performing → Adjourning",
                "Storming is normal - expect and manage conflict",
                "Active listening: Focus, clarify, reflect",
                "Conflict resolution: Collaboration = win-win",
                "Psychological safety: Safe to take risks, admit mistakes",
                "Diversity improves problem-solving",
              ],
              diagrams: [
                {
                  title: "Tuckman's Team Development Model",
                  description: "Draw stages as arrow or steps: Forming (polite, orientation) → Storming (conflict, power struggles) → Norming (cohesion, agreement) → Performing (high productivity) → Adjourning (dissolution). Show performance increasing over stages.",
                  imageUrl: "/diagrams/tuckman_model.svg",
                },
              ],
              problems: [
                {
                  question: "Explain Tuckman's stages of team development. (5 marks)",
                  answer: "Tuckman's Team Development Stages:\n\n1. FORMING:\n- Team members meet\n- Polite, guarded behavior\n- Roles and responsibilities unclear\n- Dependence on leader\n- Testing boundaries\n\n2. STORMING:\n- Conflicts emerge\n- Personality clashes\n- Competition for roles\n- Frustration with progress\n- Critical stage - many teams fail here\n\n3. NORMING:\n- Resolution of conflicts\n- Rules and norms established\n- Cooperation increases\n- Roles clarify\n- Trust develops\n\n4. PERFORMING:\n- High productivity\n- Team works smoothly\n- Focus on goals\n- Mutual support\n- Autonomous decision-making\n\n5. ADJOURNING:\n- Project completion\n- Team dissolution\n- Recognition of achievements\n- Emotional responses (sadness/relief)\n\nNote: Teams may cycle back to earlier stages with changes.",
                },
                {
                  question: "What is psychological safety? Why is it important for teams? (5 marks)",
                  answer: "Psychological Safety:\n\nDEFINITION:\nA shared belief that the team is safe for interpersonal risk-taking. Members feel comfortable being themselves and speaking up without fear of punishment or embarrassment.\n\nCHARACTERISTICS:\n- Can admit mistakes openly\n- Can ask questions without ridicule\n- Can share ideas freely\n- Can challenge status quo\n- Can be vulnerable\n\nIMPORTANCE FOR TEAMS:\n\n1. BETTER PROBLEM-SOLVING:\n- All ideas are heard\n- Problems surfaced early\n- No hidden issues\n\n2. INCREASED INNOVATION:\n- Risk-taking encouraged\n- Experimentation valued\n- Creative ideas emerge\n\n3. HIGHER ENGAGEMENT:\n- Members feel valued\n- Higher job satisfaction\n- Lower turnover\n\n4. LEARNING CULTURE:\n- Mistakes = learning opportunities\n- Continuous improvement\n- Knowledge sharing\n\nGoogle Project Aristotle found psychological safety was #1 factor in high-performing teams.",
                },
              ],
              examTips: [
                "Memorize Tuckman's 5 stages in order",
                "Storming is normal - don't avoid it",
                "Know conflict resolution approaches",
                "Psychological safety = Google's top team factor",
              ],
            },
          },
        ],
      },
    ],
  },
}
