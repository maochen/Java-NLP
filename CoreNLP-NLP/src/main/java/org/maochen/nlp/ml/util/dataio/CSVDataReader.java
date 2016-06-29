package org.maochen.nlp.ml.util.dataio;

import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by mguan on 4/8/16.
 */
public class CSVDataReader {

    private String filename;
    private int labelCol;
    private String delim;

    private boolean hasHeader;
    private String[] header;

    private Set<Integer> ignoredColumns = new HashSet<>();
    private int posNegIndex = -1; // Column determine pos or neg example.

    public List<Tuple> read() throws IOException {
        FileInputStream fileInputStream = new FileInputStream(filename);
        return read(fileInputStream);
    }

    /**
     * only extract feats, wont do label recognition or posneg recog.
     */
    private void extractValuedFeat(String[] fields, Tuple tuple, int labelCol) {
        FeatNamedVector vector = (FeatNamedVector) tuple.vector;
        for (int i = 0; i < fields.length; i++) {
            if (i == labelCol || ignoredColumns.contains(i)) {
                continue;
            }

            if (hasHeader) {
                vector.featsName[i] = header[i];
            }

            try {
                double val = Double.parseDouble(fields[i]);
                tuple.vector.getVector()[i] = val;
            } catch (NumberFormatException e) {
                tuple.vector.getVector()[i] = 1;
            }
        }

    }

    public List<Tuple> read(InputStream is) throws IOException {
        List<Tuple> ds = new ArrayList<>();

        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        String line = br.readLine();

        int count = 0;
        while (line != null) {
            if (count == 0 && hasHeader) {
                header = line.split(delim);
            } else {
                String[] values = line.split(delim);
                final int actualLabelCol = labelCol == -1 ? values.length - 1 : labelCol;

                FeatNamedVector featNamedVector = new FeatNamedVector(new double[values.length - 1 - ignoredColumns.size()]);
                Tuple tuple = new Tuple(featNamedVector);
                tuple.label = values[actualLabelCol];

                if (posNegIndex > -1) {
                    tuple.isPosExample = Integer.parseInt(values[posNegIndex]) > -1;
                }

                extractValuedFeat(values, tuple, actualLabelCol);
                ds.add(tuple);
            }

            count++;
            line = br.readLine();
        }

        return ds;
    }

    public String getHeader() {
        return Arrays.stream(this.header).collect(Collectors.joining(delim));
    }

    public CSVDataReader(String filename, int labelCol, String delim, boolean hasHeader, Set<Integer> ignoredColumns, int posNegIndex) {
        this.filename = filename;
        this.labelCol = labelCol;
        this.delim = delim;
        this.hasHeader = hasHeader;
        this.posNegIndex = posNegIndex;
        if (ignoredColumns != null) {
            this.ignoredColumns = ignoredColumns;
        }
    }
}
