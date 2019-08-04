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


    def preModelTeach(self):
        print('pre model teach')

    def postModelTeach(self):
        print('post model taught')


class TeacherEnhanced:

    def __init__(self,
                 dataBunch,
                 lossFunction,
                 trainingSubscriber: Subscriber,
                 validationSubscriber: Subscriber):
        self._dataBunch = dataBunch
        self._lossFunction = lossFunction
        self._trainingSubscriber = trainingSubscriber
        self._validationSubscriber = validationSubscriber

    def teachModel(self, model, numberOfEpochs):
        self._notifiyPreTeach()
        for epoch in range(numberOfEpochs):
            self._trainModel(model,
                             epoch)
            self._validateModel(model,
                                epoch)
        self._notifiyPostTaught()

    def _notifiyPreTeach(self):
        self._trainingSubscriber.preModelTeach()
        self._validationSubscriber.preModelTeach()

    def _notifiyPostTaught(self):
        self._trainingSubscriber.postModelTeach()
        self._validationSubscriber.postModelTeach()

    def _trainModel(self, model, epoch):
        self._processData(model,
                          self._dataBunch.trainingData,
                          epoch,
                          self._trainingSubscriber.dataProcessingEvents)

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
