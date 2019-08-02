
class TeacherEnhanced():

    def __init__(self):
        print("Constructed")



    def teachModel(self, model, dataBunch, numberOfEpochs):
        for epoch in range(numberOfEpochs):
            self._trainModel(model,
                             dataBunch.trainingData,
                             epoch)
            self._validateModel(model,
                                dataBunch.validationData,
                                epoch)

    def _trainModel(self, model, testData, epoch):
        # Train Model
        print("training")

    def _validateModel(self, model, testData, epoch):
        # Train Model
        print("validating")

