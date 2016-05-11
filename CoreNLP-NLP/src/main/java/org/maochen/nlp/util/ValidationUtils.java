package org.maochen.nlp.util;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;

import java.util.Arrays;
import java.util.List;

/**
 * Created by Maochen on 2/22/16.
 */
public class ValidationUtils {
    public static void printSequenceTuple(final SequenceTuple st, List<String> correctTag) {
        String[] tokens = st.entries.stream()
                .map(tuple -> Arrays.stream(((FeatNamedVector) tuple.vector).featsName)
                        .filter(x -> x.startsWith("w0="))
                        .map(t -> t.split("=")[1])
                        .findFirst().orElse(StringUtils.EMPTY))
                .toArray(String[]::new);

        String[] pos = st.entries.stream()
                .map(tuple -> Arrays.stream(((FeatNamedVector) tuple.vector).featsName)
                        .filter(x -> x.startsWith("pos0="))
                        .map(t -> t.split("=")[1])
                        .findFirst().orElse(StringUtils.EMPTY))
                .toArray(String[]::new);

        for (int i = 0; i < tokens.length; i++) {
            String out = tokens[i] + "\t" + pos[i] + "\t" + st.entries.get(i).label;
            if (correctTag != null && !st.entries.get(i).label.equals(correctTag.get(i))) {
                out += "\t" + "Expected:\t" + correctTag.get(i);
            }
            System.out.println(out);
        }
    }

}
