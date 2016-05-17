package org.maochen.nlp.ml.classifier.libsvm;

import libsvm.*;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.NotImplementedException;
import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.LabelIndexer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.Charset;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Created by mguan on 5/11/16.
 */
public class LibSVMClassifier implements IClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(LibSVMClassifier.class);


    private svm_model model = null;
    public svm_parameter para = null; // TODO: Evil here. change to private and so ...

    private LabelIndexer labelIndexer = null;

    private void writeToLog() {
        svm.svm_set_print_string_function(x -> {
            if (!".".equals(x)) {
                LOG.info(x);
            }
        });
    }

    public svm_parameter getDefaultPara() {
        svm_parameter para = new svm_parameter();
        para.probability = 1;
        para.gamma = 0.5;
        para.nu = 0.5;
        para.C = 100;

        para.svm_type = svm_parameter.C_SVC;
        para.kernel_type = svm_parameter.LINEAR;
        para.cache_size = 20000;
        para.eps = 0.001;
        para.p = 0.1;

        return para;
    }

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        if (para == null) {
            LOG.warn("Parameter is null. Use the default parameter.");
            this.para = getDefaultPara();
        }

        labelIndexer = new LabelIndexer(trainingData);

        svm_problem prob = new svm_problem();

        int featSize = trainingData.iterator().next().vector.getVector().length;
        prob.l = trainingData.size();
        prob.y = new double[prob.l];
        prob.x = new svm_node[prob.l][featSize];

        for (int i = 0; i < trainingData.size(); i++) {
            Tuple tuple = trainingData.get(i);
            prob.x[i] = new svm_node[featSize];

            for (int j = 0; j < tuple.vector.getVector().length; j++) {
                svm_node node = new svm_node();
                node.index = j;
                node.value = tuple.vector.getVector()[j];
                prob.x[i][j] = node;
            }

            prob.y[i] = labelIndexer.getIndex(tuple.label);
        }

        model = svm.svm_train(prob, para);
        return this;
    }

    @Override
    public Map<String, Double> predict(Tuple predict) {
        double[] feats = predict.vector.getVector();
        svm_node[] svmfeats = new svm_node[feats.length];

        for (int i = 0; i < feats.length; i++) {
            svm_node svmfeatI = new svm_node();
            svmfeatI.index = i;
            svmfeatI.value = feats[i];
            svmfeats[i] = svmfeatI;
        }

        int totalSize = labelIndexer.getLabelSize();
        int[] labels = new int[totalSize];
        svm.svm_get_labels(model, labels);

        double[] probs = new double[totalSize];
        svm.svm_predict_probability(model, svmfeats, probs);

        Map<String, Double> result = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            result.put(labelIndexer.getLabel(labels[i]), probs[i]);
        }

        return result;
    }

    @Override
    public void setParameter(Properties props) {
        throw new NotImplementedException("Use direct set para for now.");
    }

    @Override
    public void persistModel(String modelFile) throws IOException {
        if (this.labelIndexer == null) {
            throw new RuntimeException("LabelIndexer is null!");
        }

        ZipOutputStream zipos = new ZipOutputStream(new FileOutputStream(modelFile));

        String svmModelAbsolutePath = modelFile + ".model";
        String svmModelFilename = new File(svmModelAbsolutePath).getName();
        svm.svm_save_model(svmModelAbsolutePath, model);
        ZipEntry libSVMModelZipEntry = new ZipEntry(svmModelFilename);
        zipos.putNextEntry(libSVMModelZipEntry);
        IOUtils.copy(new FileInputStream(svmModelAbsolutePath), zipos);
        zipos.closeEntry();

        String labelIndexerAbsolutePath = modelFile + ".lbindexer";
        String labelIndexerFileName = new File(labelIndexerAbsolutePath).getName();
        String labelIndexerString = this.labelIndexer.serializeToString();
        ZipEntry labelIndexerZipEntry = new ZipEntry(labelIndexerFileName);
        zipos.putNextEntry(labelIndexerZipEntry);
        IOUtils.write(labelIndexerString, zipos, Charset.defaultCharset());
        zipos.closeEntry();

        IOUtils.closeQuietly(zipos);
        FileUtils.forceDelete(new File(svmModelAbsolutePath));
    }

    // We load twice because svm.svm_load_model will close the stream after load. so the next guy will have Steam closed exception.
    @Override
    public void loadModel(InputStream modelIs) {

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try {
            IOUtils.copy(modelIs, baos);
        } catch (IOException e) {
            LOG.error("Load model err.", e);
        }

        InputStream isForSVMLoad = new ByteArrayInputStream(baos.toByteArray());
        try (ZipInputStream zipInputStream = new ZipInputStream(isForSVMLoad)) {
            ZipEntry entry;
            while ((entry = zipInputStream.getNextEntry()) != null) {
                if (entry.getName().endsWith(".model")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zipInputStream, Charset.defaultCharset()));
                    this.model = svm.svm_load_model(br);
                }
            }

        } catch (IOException e) {
            // Do Nothing.
        }

        modelIs = new ByteArrayInputStream(baos.toByteArray());
        try (ZipInputStream zipInputStream = new ZipInputStream(modelIs)) {
            ZipEntry entry;
            while ((entry = zipInputStream.getNextEntry()) != null) {
                if (entry.getName().endsWith(".lbindexer")) {
                    String lbIndexer = IOUtils.toString(zipInputStream, Charset.defaultCharset());
                    this.labelIndexer = new LabelIndexer(new ArrayList<>());
                    this.labelIndexer.readFromSerializedString(lbIndexer);
                }
            }

        } catch (IOException e) {
            LOG.error("Err in load LabelIndexer", e);
        }

    }

    public LibSVMClassifier() {
        super();
        writeToLog();
    }

}
