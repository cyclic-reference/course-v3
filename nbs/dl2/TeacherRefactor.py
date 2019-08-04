class Subscriber:
    def __init__(self):
        print('finna bust a nut')

    def postEpoch(self, epochNumber):
        print('post epoch')

    def postBatchEvaluation(self, loss):
        print('post batch eval')

    def preBatchEvaluation(self):
        print('I\'m gonna pre')

    def preEpoch(self, epoch):
        print('pre epoch')

    def preModelTeach(self, model, epochs):
        print('pre model teach')

    def postModelTeach(self):
        print('post model taught')

class TrainingSubscriber(Subscriber):

    def __init__(self):
        print('Training subscriber init')

    def postEpoch(self, epochNumber):
        print('training post epoch')

    def preBatchEvaluation(self):
        print('training pre evaluation')


class ValidationSubscriber(Subscriber):

    def __init__(self):
        print('Validation subscriber init')

    def postEpoch(self, epochNumber):
        print('Validation post epoch')


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
                          self._dataBunch.trainingData,
                          epoch,
                          self._trainingSubscriber)

    def _validateModel(self, model, epoch):
        with torch.no_grad():
            self._processData(model,
                              self._dataBunch.validationData,
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
