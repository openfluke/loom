package testing

import "testing"

func TestCNN1TrainingMatrix(t *testing.T) {
	if !RunGenericLayerSuite(cnn1Spec, TestTraining) {
		t.Fatal("CNN1 training matrix had failures")
	}
}

func TestMHAForwardParity(t *testing.T) {
	if !RunGenericLayerSuite(mhaSpec, TestForward) {
		t.Fatal("MHA forward parity had failures")
	}
}

func TestMHATrainingMatrix(t *testing.T) {
	if !RunGenericLayerSuite(mhaSpec, TestTraining) {
		t.Fatal("MHA training matrix had failures")
	}
}
