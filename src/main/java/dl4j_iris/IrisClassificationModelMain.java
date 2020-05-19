package dl4j_iris;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class IrisClassificationModelMain {
    static double learningRate = 0.001;
    static int batchSize = 1;
    static int nEpochs = 200;
    static int numIn = 4;
    static int numOut = 3;
    static int nHidden = 10;
    static int classIndex = 4;
    static String filePathTrain;
    static String filePathTest;

    public static void main(String[] args) throws Exception {

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numIn)
                        .nOut(nHidden)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(nHidden)
                        .nOut(numOut)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();

        filePathTrain = new ClassPathResource("iris_train.csv").getFile().getPath();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filePathTrain)));
        DataSetIterator dataSetTrain = new RecordReaderDataSetIterator(rr, batchSize, classIndex, numOut);
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        //Serveur DeepLearning4j
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        // Entrainement du modele
        System.out.println("----------- Entrainement du modele -----------------");
        for (int i = 1; i <= nEpochs; i++) {
            System.out.println("Epoque : " + (i));
            model.fit(dataSetTrain);
        }
        // Evaluation du modele
        System.out.println("----------- Evaluation du modele -----------------");
        filePathTest = new ClassPathResource("iris_test.csv").getFile().getPath();
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filePathTest)));
        DataSetIterator dataSetTest = new RecordReaderDataSetIterator(rrTest, batchSize, classIndex, numOut);
        Evaluation evaluation = new Evaluation(numOut);
        while (dataSetTest.hasNext()) {
            DataSet dataSet = dataSetTest.next();
            INDArray features = dataSet.getFeatures();
            INDArray labels = dataSet.getLabels();
            INDArray predicted = model.output(features);
            evaluation.eval(labels, predicted);
        }
//        Staitsiques
        System.out.println(evaluation.stats());


//        System.out.println("Prediction....");
//		INDArray inputData = Nd4j.create(new double[][] {
//			{5.1,3.5,1.4,0.2},
//			{4.9,3.0,1.4,0.2},
//			{6.7,3.1,4.4,1.4},
//			{5.6,3.0,4.5,1.5},
//			{6.0,3.0,4.8,1.8},
//			{6.9,3.1,5.4,2.1},
//		});
//
//		INDArray output = model.output(inputData);
//	ModelSerializer.writeModel(model, "mymodel.model",true);
//		int[] classes=output.argMax(1).toIntVector();
//
//		System.out.println(output);
//		for(int i = 0; i < classes.length; i++) {
//			System.out.println("classe : " + llabels[classes[i]] );
//		}
    }

}
