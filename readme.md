# CartGo
CartGo is a Go package for building and using CART (Classification and Regression Tree) models for decision-making tasks. It allows users to train models on datasets, make predictions, and save/load models for later use.

## Installation
To install CartGo, use the go get command:

```
go get github.com/abedinia/cartgo/cart
```
This will download the CartGo package along with its dependencies.

##  Usage
Training a Model
To train a CART model with your dataset, first prepare your data as slices of float64 for both features (X) and labels (y):

```go
package main

import (
    "fmt"
	"github.com/abedinia/cartgo/cart"
)

func main() {
    X := [][]float64{{1, 2}, {3, 4}, {5, 6}}
    y := []float64{0, 1, 0}
    
    decision_tree := cart.NewCART()
	decision_tree.Fit(X, y)
    fmt.Println("CART model trained.")
}
```

### Making Predictions
Once your model is trained, you can use it to make predictions on new data:

```go
testX := [][]float64{{1, 2}, {3, 4}, {5, 6}}
predictions := cart.Predict(testX)
fmt.Println("Predictions:", predictions)
```

## Saving and Loading Models
You can save your trained model to a file and load it later:

```go
err := cart.SaveModel("cart_model.gob")
if err != nil {
fmt.Println("Failed to save the model:", err)
return
}
fmt.Println("Model saved successfully.")

loadedCart, err := cart.LoadModel("cart_model.gob")
if err != nil {
fmt.Println("Failed to load the model:", err)
return
}
fmt.Println("Model loaded successfully.")
```

### License
Apache-2.0 license