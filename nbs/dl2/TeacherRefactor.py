class Subscriber:
    def __init__(self):
        self._totalEpochs = 0
        self._currentEpoch = 0

    def postEpoch(self, epochNumber):
        pass

    def postBatchEvaluation(self, predictions, validationData):
        pass

    def preBatchEvaluation(self):
        pass

    def preEpoch(self, epoch, dataLoader):
        self._currentEpoch = epoch
        pass

    def preModelTeach(self, model, epochs):
        self._totalEpochs = epochs
        pass

    def postModelTeach(self):
        pass


class StatisticsSubscriber(Subscriber):

    def __init__(self,
                 accuracyFunction=accuracy,
                 name="Steve"):
        super().__init__()
        self._epochAccuracy = 0.
        self._epochLoss = 0.
        self._numberOfBatches = 0
        self._accuracyFunction = accuracyFunction
        self._name = name

    def preEpoch(self, epoch, dataLoader):
        super().preEpoch(epoch, dataLoader)
        self._numberOfBatches = len(dataLoader)
        self._epochAccuracy, self._epochLoss = 0., 0.

    def postBatchEvaluation(self, predictions, validationData):
        super().postBatchEvaluation(predictions, validationData)
        self._epochAccuracy += self._accuracyFunction(predictions, validationData)

    def postBatchLossConsumption(self, loss):
        self._epochLoss += loss

    def postEpoch(self, epochNumber):
        super().postEpoch(epochNumber)
        print("Epoch #{} {}: Loss {} Accuracy {}".format(epochNumber,
                                                         self._name,
                                                         self._epochLoss / self._numberOfBatches,
                                                         self._epochAccuracy / self._numberOfBatches))


class TrainingSubscriber(StatisticsSubscriber):

    def __init__(self,
                 lossFunction=Functional.cross_entropy,
                 schedulingFunctions=[cosineScheduler(1e-1, 1e-6), cosineScheduler(1e-1, 1e-6)], ):
        super().__init__(name="Training")
        self._optimizer = None
        self._schedulingFunctions = schedulingFunctions
        self._lossFunction = lossFunction

    def preModelTeach(self, model, epochs):
        super().preModelTeach(model, epochs)
        self._optimizer = optim.SGD(model.parameters(), self._schedulingFunctions[0](0))
        self._totalEpochs = epochs

    def postBatchEvaluation(self, predictions, valdationData):
        super().postBatchEvaluation(predictions, valdationData)
        calculatedLoss = self._lossFunction(predictions, valdationData)
        self._teachModel(calculatedLoss)
        self.postBatchLossConsumption(calculatedLoss)

    def _teachModel(self, loss):
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

    def preBatchEvaluation(self):
        super().preBatchEvaluation()
        self._annealLearningRate()

    def _annealLearningRate(self):
        for parameterGroup, schedulingFunction in zip(self._optimizer.param_groups, self._schedulingFunctions):
            parameterGroup['lr'] = schedulingFunction(self._currentEpoch / self._totalEpochs)


class ValidationSubscriber(StatisticsSubscriber):

    def __init__(self):
        super().__init__(name="Validation")


class TeacherEnhanced:
    def __init__(self,
                 dataBunch,
                 trainingSubscriber: TrainingSubscriber,
                 validationSubscriber: ValidationSubscriber):
        self._dataBunch = dataBunch
        self._trainingSubscriber = trainingSubscriber
        self._validationSubscriber = validationSubscriber

    def teachModel(self, model, numberOfEpochs):
        self._notifiyPreTeach(model, numberOfEpochs)
        for epoch in range(numberOfEpochs):
            self._trainModel(model,
                             epoch)
            self._validateModel(model,
                                epoch)
        self._notifiyPostTaught()

    def _notifiyPreTeach(self, model, epochs):
        self._trainingSubscriber.preModelTeach(model, epochs)
        self._validationSubscriber.preModelTeach(model, epochs)

    def _notifiyPostTaught(self):
        self._trainingSubscriber.postModelTeach()
        self._validationSubscriber.postModelTeach()

    def _trainModel(self, model, epoch):
        self._processData(model,
                          self._dataBunch.trainingDataSet,
                          epoch,
                          self._trainingSubscriber)

    def _validateModel(self, model, epoch):
        with torch.no_grad():
            self._processData(model,
                              self._dataBunch.validationDataSet,
                              epoch,
                              self._validationSubscriber)

    def _processData(self,
                     model,
                     dataLoader,
                     epoch,
                     processingSubscriber: Subscriber):
        processingSubscriber.preEpoch(epoch, dataLoader)
        for _xDataBatch, _yDataBatch in dataLoader:
            processingSubscriber.preBatchEvaluation()
            _predictions = model(_xDataBatch)
            processingSubscriber.postBatchEvaluation(_predictions, _yDataBatch)
        processingSubscriber.postEpoch(epoch)
