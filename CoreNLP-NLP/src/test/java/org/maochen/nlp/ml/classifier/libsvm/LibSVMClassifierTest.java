package org.maochen.nlp.ml.classifier.libsvm;

import com.google.common.collect.Lists;
import org.apache.commons.io.FileUtils;
import org.junit.Test;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.LabelIndexer;
import org.maochen.nlp.ml.util.dataio.CSVDataReader;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Created by mguan on 5/11/16.
 */
public class LibSVMClassifierTest {


    private Object getField(Object instance, String fieldName) throws NoSuchFieldException,
            ClassNotFoundException, IllegalAccessException {
        Class cls = Class.forName("org.maochen.nlp.ml.classifier.libsvm.LibSVMClassifier");
        Field dataField = cls.getDeclaredField(fieldName);
        dataField.setAccessible(true);
        return dataField.get(instance);
    }

    @Test
    public void testReadWriteModel() throws IOException, IllegalAccessException, NoSuchFieldException, ClassNotFoundException {
        Tuple oneData = new Tuple(new double[]{1, 2});
        oneData.label = "Like";

        Tuple twoData = new Tuple(new double[]{8, 7});
        twoData.label = "DisLike";


        LibSVMClassifier libSVMClassifier = new LibSVMClassifier();
        libSVMClassifier.train(Lists.newArrayList(oneData, twoData));
        assertNotNull(getField(libSVMClassifier, "model"));

        LabelIndexer labelIndexer = (LabelIndexer) getField(libSVMClassifier, "labelIndexer");
        assertNotNull(labelIndexer);
        assertEquals(2, labelIndexer.getLabelSize());
    }

    @Test
    public void testOnStudentExam() throws IOException {
        InputStream trainIs = this.getClass().getResourceAsStream("/training_data/student_exam_data.txt");

        List<Tuple> trainingData = new CSVDataReader(null, -1, ",", null, -1).read(trainIs);

        LibSVMClassifier libSVMClassifier = new LibSVMClassifier();
        libSVMClassifier.train(trainingData);

        Path tempFile = Files.createTempFile(null, null);

        libSVMClassifier.persistModel(tempFile.toAbsolutePath().toString());

        libSVMClassifier = new LibSVMClassifier();
        libSVMClassifier.loadModel(FileUtils.openInputStream(new File(tempFile.toAbsolutePath().toString())));


        int wrongCt = 0;
        for (Tuple trainTuple : trainingData) {

            String realLabel = libSVMClassifier.predict(trainTuple).entrySet().stream().max(
                    (e1, e2) -> Double.compare(e1.getValue(), e2.getValue())
            ).map(Map.Entry::getKey).orElse(null);

            if (!trainTuple.label.equals(realLabel)) {
                wrongCt++;
            }
        }

        double precision = (1 - (wrongCt / (double) trainingData.size()));

        assertEquals(0.89, precision, 0.15);
    }
}
