package org.maochen.nlp.ml.util.dataio;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by mguan on 4/8/16.
 */
public class CSVDataReader {
    @SuppressWarnings("unused")
    private static final Logger LOG = LoggerFactory.getLogger(CSVDataReader.class);

    private String filename;
    private String delim;

    int labelCol;

    String[] header = null;

    Set<Integer> ignoredColumns = new HashSet<>();
    int posNegIndex = -1; // Column determine pos or neg example.

    public List<Tuple> read() throws IOException {
        FileInputStream fileInputStream = new FileInputStream(filename);
        return read(fileInputStream);
    }

    protected Tuple extractValuedFeat(CSVRecord record) {
        FeatNamedVector featNamedVector = new FeatNamedVector(new double[record.size() - 1 - ignoredColumns.size()]);
        featNamedVector.featsName = new String[record.size()];

        Tuple tuple = new Tuple(featNamedVector);
        tuple.label = record.get(labelCol);

        for (int i = 0; i < record.size(); i++) {
            if (i == labelCol || ignoredColumns.contains(i)) {
                continue;
            }

            featNamedVector.featsName[i] = header[i];

            try {
                double val = Double.parseDouble(record.get(i));
                tuple.vector.getVector()[i] = val;
            } catch (NumberFormatException e) {
                if (header != null) {
                    featNamedVector.featsName[i] += "_" + record.get(i).toLowerCase().trim();
                }

                double val = record.get(i).trim().isEmpty() ? 0 : 1;
                tuple.vector.getVector()[i] = val;
            }
        }

        return tuple;
    }

    public List<Tuple> read(InputStream is) throws IOException {
        CSVFormat format = CSVFormat.RFC4180.withHeader().withDelimiter(delim.charAt(0));
        CSVParser csvParser = new CSVParser(new InputStreamReader(is), format);

        List<CSVRecord> records = csvParser.getRecords();
        header = csvParser.getHeaderMap().entrySet().stream()
                .sorted((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
                .map(Map.Entry::getKey).toArray(String[]::new);

        labelCol = labelCol == -1 ? records.get(0).size() - 1 : labelCol;

        List<Tuple> ds = records.stream().parallel().map(this::extractValuedFeat).collect(Collectors.toList());
        return ds;
    }

    public CSVDataReader(String filename, int labelCol, String delim, Set<Integer> ignoredColumns, int posNegIndex) {
        this.filename = filename;
        this.labelCol = labelCol;
        this.delim = delim;
        this.posNegIndex = posNegIndex;
        if (ignoredColumns != null) {
            this.ignoredColumns = ignoredColumns;
        }
    }
}
