package classifier.backcup;

import java.util.List;

import opennlp.maxent.DataStream;

public class ListDataStream implements DataStream {
    List<String[]> trainData;
    int iter;

    public ListDataStream(List<String[]> trainData) {
        this.trainData = trainData;
        this.iter = 0;
    }

    @Override
    public Object nextToken() {
        String[] entry = trainData.get(iter++);
        StringBuffer buf = new StringBuffer();
        for (String s : entry) {
            buf.append(s);
            buf.append(" ");
        }
        return buf.toString().trim();
    }

    @Override
    public boolean hasNext() {
        if (iter == trainData.size() - 1) return false;
        return true;
    }

}
