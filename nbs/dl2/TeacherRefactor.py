class Subscriber:
    def __init__(self):
        pass

    def postEpoch(self, epochNumber):
        pass

    def postBatchEvaluation(self, loss):
        pass

    def preBatchEvaluation(self):
        pass

    def preEpoch(self, epoch):
        pass

    def preModelTeach(self, model, epochs):
        pass

    def postModelTeach(self):
        pass

class TrainingSubscriber(Subscriber):

    def __init__(self,
                 schedulingFunctions=[cosineScheduler(1e-1, 1e-6), cosineScheduler(1e-1, 1e-6)]):
        print('Training subscriber init')
        self._optimizer = None
        self._totalEpochs = 0
        self._schedulingFunctions = schedulingFunctions

    def postEpoch(self, epochNumber):
        pass

    def preModelTeach(self, model, epochs):
        self._optimizer = optim.SGD(model.parameters(), self._schedulingFunctions[0](0))
        self._totalEpochs = epochs

    def postBatchEvaluation(self, loss):
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

    def preBatchEvaluation(self):
        pass


class ValidationSubscriber(Subscriber):

    def __init__(self):
        pass

    def postEpoch(self, epochNumber):
        pass


class TeacherEnhanced:
    def __init__(self,
                 dataBunch,
                 lossFunction,
                 trainingSubscriber: TrainingSubscriber,
                 validationSubscriber: ValidationSubscriber):
        self._dataBunch = dataBunch
        self._lossFunction = lossFunction
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
                     dataProcessingSubscriber: Subscriber):
        dataProcessingSubscriber.preEpoch(epoch)
        for _xDataBatch, _yDataBatch in dataLoader:
            dataProcessingSubscriber.preBatchEvaluation()
            _predictions = model(_xDataBatch)
            calculatedLoss = self._lossFunction(_predictions, _yDataBatch)
            dataProcessingSubscriber.postBatchEvaluation(calculatedLoss)
        dataProcessingSubscriber.postEpoch(epoch)
