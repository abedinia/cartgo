package cart

import (
	"os"
	"reflect"
	"testing"
)

func TestGini(t *testing.T) {
	tests := []struct {
		y        []float64
		expected float64
	}{
		{[]float64{1, 1, 1, 1}, 0.0},
		{[]float64{1, 0, 1, 0}, 0.5},
		{[]float64{1, 2, 1, 2, 3}, 0.64},
	}

	for _, test := range tests {
		got := Gini(test.y)

		epsilon := 0.0001
		if diff := abs(got - test.expected); diff > epsilon {
			t.Errorf("Gini(%v) = %v; want %v (within epsilon: %v)", test.y, got, test.expected, epsilon)
		}
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestSplitDataset(t *testing.T) {
	X := [][]float64{
		{1, 2}, {3, 4}, {5, 6},
	}
	y := []float64{0, 1, 0}
	featureIndex := 0
	threshold := 4.0

	leftX, rightX, leftY, rightY := splitDataset(X, y, featureIndex, threshold)

	expectedLeftX := [][]float64{{1, 2}, {3, 4}}
	expectedRightX := [][]float64{{5, 6}}
	expectedLeftY := []float64{0, 1}
	expectedRightY := []float64{0}

	if !reflect.DeepEqual(leftX, expectedLeftX) ||
		!reflect.DeepEqual(rightX, expectedRightX) ||
		!reflect.DeepEqual(leftY, expectedLeftY) ||
		!reflect.DeepEqual(rightY, expectedRightY) {
		t.Errorf("splitDataset did not split as expected")
	}
}

func TestPredictSingle(t *testing.T) {
	root := &DecisionNode{
		FeatureIndex: 0,
		Threshold:    1.5,
		Left: &DecisionNode{
			Value:  0.0,
			IsLeaf: true,
		},
		Right: &DecisionNode{
			Value:  1.0,
			IsLeaf: true,
		},
	}
	cart := &CART{Root: root}
	X := [][]float64{
		{1.0}, {2.0},
	}
	expected := []float64{0.0, 1.0}
	got := cart.Predict(X)

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("Predict() = %v; want %v", got, expected)
	}
}

func TestPredictComplexTree(t *testing.T) {

	root := &DecisionNode{
		FeatureIndex: 1,
		Threshold:    3.5,
		Left: &DecisionNode{
			FeatureIndex: 0,
			Threshold:    2.5,
			Left: &DecisionNode{
				Value:  0.0,
				IsLeaf: true,
			},
			Right: &DecisionNode{
				Value:  1.0,
				IsLeaf: true,
			},
		},
		Right: &DecisionNode{
			Value:  2.0,
			IsLeaf: true,
		},
	}
	cart := &CART{Root: root}

	X := [][]float64{
		{1.0, 4.0},
		{3.0, 2.0},
	}
	expected := []float64{2.0, 1.0}
	got := cart.Predict(X)

	if !reflect.DeepEqual(got, expected) {
		t.Errorf("Predict() = %v; want %v", got, expected)
	}
}

func TestSaveLoadModel(t *testing.T) {

	cart := NewCART()
	X := [][]float64{{1, 2}, {3, 4}}
	y := []float64{0, 1}
	cart.Fit(X, y)

	filename := "test_cart_model.gob"
	if err := cart.SaveModel(filename); err != nil {
		t.Fatalf("SaveModel() failed: %v", err)
	}

	loadedCart, err := LoadModel(filename)
	if err != nil {
		t.Fatalf("LoadModel() failed: %v", err)
	}

	expected := cart.Predict(X)
	got := loadedCart.Predict(X)
	if !reflect.DeepEqual(got, expected) {
		t.Errorf("Loaded model predictions = %v; want %v", got, expected)
	}

	if err := os.Remove(filename); err != nil {
		t.Logf("Failed to clean up test file: %v", err)
	}
}
