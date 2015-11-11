package org.maochen.nlp.util;

import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

/**
 * Go through every document, count prob for P(w)=C(w)/C(documents), C(w) for every doc is either 0
 * or 1, not the total count in that doc. Here doc is sentence. Created by Maochen on 5/1/15.
 */
public class StopwordsGenerator {

    private Map<String, Double> wordCount = new ConcurrentHashMap<>();

    private AtomicLong totalCount = new AtomicLong(0);

    // This is based on documents/sentences.
    static class DocumentCount {
        private static void addCount(String sentence, StopwordsGenerator stopwordsGenerator) {
            sentence = stopwordsGenerator.stringNormalize(sentence);

            if (sentence.isEmpty()) { // invalid sentence.
                return;
            }

            stopwordsGenerator.totalCount.addAndGet(1); // increase the total doc count.

            Set<String> tokens = Arrays.stream(sentence.split("\\s")).parallel().collect
                    (Collectors.toSet());

            tokens.parallelStream().forEach(token -> {
                Double tokenCount = stopwordsGenerator.wordCount.containsKey(token) ?
                        stopwordsGenerator.wordCount.get(token) : 0.0D;
                tokenCount++;
                stopwordsGenerator.wordCount.put(token, tokenCount);
            });
        }


        public static void generateFromFile(File file, StopwordsGenerator stopwordsGenerator) {
            try (BufferedReader br = new BufferedReader(new FileReader(file))) {
                StringBuilder sb = new StringBuilder();
                String line = br.readLine();

                while (line != null) {
                    if (line.trim().isEmpty()) {
                        String[] sentences = sb.toString().split("\\."); // Split sentences.
                        Arrays.stream(sentences).forEach(s -> DocumentCount.addCount(s,
                                stopwordsGenerator)); // TODO: not parallelled. wordCount will
                                // miss. Need investigate
                        sb.setLength(0);
                    } else {
                        sb.append(line);
                        sb.append(StringUtils.SPACE);
                    }

                    line = br.readLine();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    // This is based on single tokens.
    static class WikiSingleWordCount {
        public static void generateFromFile(File file, StopwordsGenerator stopwordsGenerator) {
            int maxThreshold = -1;
            StringBuilder wordBuilder = new StringBuilder();

            try (BufferedReader br = new BufferedReader(new FileReader(file))) {
                int c = br.read();

                while (c != -1) {
                    if (maxThreshold > 0 && stopwordsGenerator.totalCount.get() > maxThreshold) {
                        break;
                    }

                    if (c != ' ') {
                        wordBuilder.append(((char) c));
                    } else {
                        String token = stopwordsGenerator.stringNormalize(wordBuilder.toString());
                        wordBuilder.setLength(0);
                        Double count = stopwordsGenerator.wordCount.containsKey(token) ?
                                stopwordsGenerator.wordCount.get(token) : 0.0D;
                        stopwordsGenerator.wordCount.put(token, ++count);
                        stopwordsGenerator.totalCount.addAndGet(1);
                        if (stopwordsGenerator.totalCount.get() % 10000000 == 0) {
                            if (maxThreshold > 0) {
                                System.out.println("Processed tokens: " + stopwordsGenerator
                                        .totalCount.get() / (double) maxThreshold * 100 + "%");
                            } else {
                                System.out.println("Processed tokens: " + stopwordsGenerator
                                        .totalCount.get());
                            }
                        }
                    }
                    c = br.read();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public String stringNormalize(String str) {
        str = str.replaceAll("\"", StringUtils.EMPTY) // double quotes
                .replaceAll(",", StringUtils.SPACE) // remove ,
                .replaceAll("\\p{Punct}+$", StringUtils.EMPTY) // remove punct at the end
                .replaceAll("[?:;!]", StringUtils.EMPTY) // remove char
                .replaceAll("--", StringUtils.EMPTY)
                .replaceAll("i've", "i have")
                .replaceAll("we'll", "we will")
                .replaceAll("he's", "he has")
                .replaceAll("'", StringUtils.SPACE)
                .replaceAll("\\s+", StringUtils.SPACE) // extra spacer
                .toLowerCase().trim();
        return str;
    }

    public void normalize() {
        for (String token : wordCount.keySet()) {
            Double count = wordCount.get(token);
            count /= totalCount.doubleValue();
            wordCount.put(token, count);
        }
    }

    public Map<String, Double> getProbability() {
        return wordCount;
    }

    public void writeFile(String fileName, List<Map.Entry<String, Double>> result) {
        try {
            File file = new File(fileName);
            BufferedWriter output = new BufferedWriter(new FileWriter(file));
            result.stream().forEach(entry -> {
                        try {
                            output.write(entry.toString() + System.lineSeparator());
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
            );
            output.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Give args[0] as the root folder (or just single file) that contains all documents.
    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Please specify dir or filename | output file location");
            return;
        }

        StopwordsGenerator stopwordsGenerator = new StopwordsGenerator();

        File file = new File(args[0]);

        if (file.isFile()) {
            WikiSingleWordCount.generateFromFile(file, stopwordsGenerator);
        } else {
            File[] files = file.listFiles();
            Arrays.stream(files).parallel()
                    .filter(File::isFile)
                    .forEach(f -> WikiSingleWordCount.generateFromFile(f, stopwordsGenerator));
        }

        stopwordsGenerator.normalize();
        List<Map.Entry<String, Double>> result = stopwordsGenerator.getProbability()
                .entrySet().stream()
                .sorted((o1, o2) -> o2.getValue().compareTo(o1.getValue()))
                .collect(Collectors.toList());
        stopwordsGenerator.writeFile(args[1], result);
    }
}
