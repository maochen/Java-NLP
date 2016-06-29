package org.maochen.nlp.ml.util.dataio;

import org.apache.commons.lang3.StringUtils;
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
    private String[] header = null;

    private Set<Integer> ignoredColumns = new HashSet<>();
    private int posNegIndex = -1; // Column determine pos or neg example.

    public List<Tuple> read() throws IOException {
        FileInputStream fileInputStream = new FileInputStream(filename);
        return read(fileInputStream);
    }

    protected Tuple extractValuedFeat(String[] fields, int labelCol, Set<Integer> ignoredColumns, String[] header, int posNegIndex) {
        FeatNamedVector featNamedVector = new FeatNamedVector(new double[fields.length - 1 - ignoredColumns.size()]);
        featNamedVector.featsName = new String[fields.length];

        Tuple tuple = new Tuple(featNamedVector);
        tuple.label = fields[labelCol];

        if (posNegIndex > -1) {
            tuple.isPosExample = Integer.parseInt(fields[posNegIndex]) > -1;
        }

        for (int i = 0; i < fields.length; i++) {
            if (i == labelCol || ignoredColumns.contains(i)) {
                continue;
            }

            if (header != null) {
                featNamedVector.featsName[i] = header[i];
            } else {
                featNamedVector.featsName[i] = String.valueOf(i);
            }

            try {
                double val = Double.parseDouble(fields[i]);
                tuple.vector.getVector()[i] = val;
            } catch (NumberFormatException e) {
                if (header != null) {
                    featNamedVector.featsName[i] += "_" + fields[i].toLowerCase().trim();
                }
                double val = fields[i].trim().isEmpty() ? 0 : 1;
                tuple.vector.getVector()[i] = val;
            }
        }

        return tuple;
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
                Tuple tuple = extractValuedFeat(values, actualLabelCol, ignoredColumns, header, posNegIndex);
                ds.add(tuple);
            }

            count++;
            line = br.readLine();
        }

        return ds;
    }

    public String getHeader() {
        return this.header == null ? StringUtils.EMPTY : Arrays.stream(this.header).collect(Collectors.joining(delim));
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
