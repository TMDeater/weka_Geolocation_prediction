/**
 * Created by MSI on 27-Nov-17.
 */
import weka.classifiers.Classifier;
import weka.core.*;

import java.io.BufferedReader;
import java.io.FileReader;

public class test_svm {
    public static void main(String[] args) throws Exception {
        displayPredict(predict(22.350231,114.099715));
        displayPredict(predict(22.438438,114.065052));
        displayPredict(predict(22.365595,113.934474));
        displayPredict(predict(22.414758,114.308767));
        displayPredict(predict(22.502324,114.127814));
        displayPredict(predict(22.320656,113.939613));
    }

    public static String predict(double lat, double lng) throws Exception {

        //load model
        String rootPath="C:/Users/MSI/Desktop/pg/";
        Classifier svm = (Classifier) weka.core.SerializationHelper.read(rootPath+"svm_model_1207_true.model");


        //load a data source for getting the class value attribute structure
        Instances dataSource= new Instances(
                new BufferedReader(
                        new FileReader(rootPath+"mergedData.arff")
                )
        );
        //add a new instance into the data source end
        double[] instanceValue1 = new double[dataSource.numAttributes()];
        instanceValue1[1] = lat;
        instanceValue1[2] = lng;
        dataSource.add(new DenseInstance(1.0,instanceValue1));
        dataSource.setClassIndex(0);


        //perform prediction
        int ttl = dataSource.numInstances();
        double value=svm.classifyInstance(dataSource.instance(ttl-1));

        //get the name of the class value
        return dataSource.classAttribute().value((int)value);
    }

    public static void displayPredict(String prediction){
        System.out.println("The predicted value of instance "+
                Integer.toString(0)+
                ": "+prediction);
    }
}
