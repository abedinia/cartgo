package cart

import (
	"encoding/gob"
	"os"
)

type DecisionNode struct {
	FeatureIndex int
	Threshold    float64
	Left         *DecisionNode
	Right        *DecisionNode
	Value        float64
	IsLeaf       bool
}

type CART struct {
	Root *DecisionNode
}

func NewCART() *CART {
	return &CART{}
}

func (cart *CART) Fit(X [][]float64, y []float64) {
	cart.Root = cart.buildTree(X, y, 0)
}

func (cart *CART) buildTree(X [][]float64, y []float64, depth int) *DecisionNode {
	if len(X) == 0 {
		return nil
	}
	if isHomogeneous(y) {
		return &DecisionNode{
			Value:  y[0],
			IsLeaf: true,
		}
	}

	bestFeatureIndex, bestThreshold := cart.findBestSplit(X, y)
	if bestFeatureIndex == -1 {

		return &DecisionNode{
			Value:  majorityValue(y),
			IsLeaf: true,
		}
	}

	leftX, rightX, leftY, rightY := splitDataset(X, y, bestFeatureIndex, bestThreshold)

	node := &DecisionNode{
		FeatureIndex: bestFeatureIndex,
		Threshold:    bestThreshold,
		Left:         cart.buildTree(leftX, leftY, depth+1),
		Right:        cart.buildTree(rightX, rightY, depth+1),
	}

	return node
}

func Gini(y []float64) float64 {
	var total = float64(len(y))
	if total == 0 {
		return 0
	}
	labelCounts := make(map[float64]float64)
	for _, label := range y {
		labelCounts[label]++
	}
	var impurity = 1.0
	for _, count := range labelCounts {
		probOfLabel := count / total
		impurity -= probOfLabel * probOfLabel
	}
	return impurity
}

func (cart *CART) findBestSplit(X [][]float64, y []float64) (int, float64) {
	bestImpurity := 1.0
	bestFeatureIndex := -1
	bestThreshold := 0.0
	nSamples := len(X)

	for featureIndex := 0; featureIndex < len(X[0]); featureIndex++ {
		featureValues := make(map[float64]bool)
		for _, sample := range X {
			featureValues[sample[featureIndex]] = true
		}

		for value := range featureValues {
			leftY, rightY := []float64{}, []float64{}
			for sampleIndex, sample := range X {
				if sample[featureIndex] < value {
					leftY = append(leftY, y[sampleIndex])
				} else {
					rightY = append(rightY, y[sampleIndex])
				}
			}

			impurity := (Gini(leftY)*float64(len(leftY)) + Gini(rightY)*float64(len(rightY))) / float64(nSamples)
			if impurity < bestImpurity {
				bestImpurity = impurity
				bestFeatureIndex = featureIndex
				bestThreshold = value
			}
		}
	}
	return bestFeatureIndex, bestThreshold
}

func splitDataset(X [][]float64, y []float64, featureIndex int, threshold float64) ([][]float64, [][]float64, []float64, []float64) {
	leftX, rightX, leftY, rightY := [][]float64{}, [][]float64{}, []float64{}, []float64{}
	for i, sample := range X {
		if sample[featureIndex] < threshold {
			leftX = append(leftX, sample)
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, sample)
			rightY = append(rightY, y[i])
		}
	}
	return leftX, rightX, leftY, rightY
}

func majorityValue(y []float64) float64 {
	labelCounts := make(map[float64]int)
	maxCount := 0
	var majorityLabel float64
	for _, label := range y {
		labelCounts[label]++
		if labelCounts[label] > maxCount {
			maxCount = labelCounts[label]
			majorityLabel = label
		}
	}
	return majorityLabel
}

func isHomogeneous(y []float64) bool {
	firstLabel := y[0]
	for _, label := range y[1:] {
		if label != firstLabel {
			return false
		}
	}
	return true
}

func (cart *CART) Predict(X [][]float64) []float64 {
	predictions := make([]float64, len(X))
	for i, sample := range X {
		predictions[i] = cart.predictSingle(sample, cart.Root)
	}
	return predictions
}

func (cart *CART) predictSingle(x []float64, node *DecisionNode) float64 {
	if node.IsLeaf {
		return node.Value
	}
	if x[node.FeatureIndex] < node.Threshold {
		return cart.predictSingle(x, node.Left)
	} else {
		return cart.predictSingle(x, node.Right)
	}
}

func init() {
	gob.Register(&CART{})
	gob.Register(&DecisionNode{})
}

func (cart *CART) SaveModel(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(cart)
	if err != nil {
		return err
	}

	return nil
}

func LoadModel(filename string) (*CART, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var cart CART
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&cart)
	if err != nil {
		return nil, err
	}

	return &cart, nil
}
