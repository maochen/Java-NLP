package org.maochen.utils;

import org.apache.commons.lang3.StringUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Go through every document, count prob for P(w)=C(w)/C(documents), C(w) for every doc is either 0 or 1, not the total count in that doc.
 * Here doc is sentence.
 * Created by Maochen on 5/1/15.
 */
public class StopwordsGenerator {

    private Map<String, Double> wordCount = new ConcurrentHashMap<>();

    private AtomicInteger docCount = new AtomicInteger(0);

    public void addCount(String sentence) {
        sentence = sentence.trim()
                .replaceAll("\"", StringUtils.EMPTY) // double quotes
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

        if (sentence.isEmpty()) { // invalid sentence.
            return;
        }

        docCount.addAndGet(1); // increase the total doc count.

        Set<String> tokens = Arrays.stream(sentence.split("\\s")).parallel().collect(Collectors.toSet());

        tokens.parallelStream().forEach(token -> {
            Double tokenCount = wordCount.containsKey(token) ? wordCount.get(token) : 0.0D;
            tokenCount++;
            wordCount.put(token, tokenCount);
        });
    }

    public void normalize() {
        for (String token : wordCount.keySet()) {
            Double count = wordCount.get(token);
            count /= docCount.doubleValue();
            wordCount.put(token, count);
        }
    }

    public Map<String, Double> getProbability() {
        return wordCount;
    }

    public void generateFromFile(File file) {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();

            while (line != null) {
                if (line.trim().isEmpty()) {
                    String[] sentences = sb.toString().split("\\."); // Split sentences.
                    Arrays.stream(sentences).forEach(this::addCount); // TODO: not parallelled. wordCount will miss. Need investigate
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

    // Give args[0] as the root folder (or just single file) that contains all documents.
    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Please specify dir or filename");
            return;
        }

        StopwordsGenerator g = new StopwordsGenerator();

        File file = new File(args[0]);

        if (file.isFile()) {
            g.generateFromFile(file);
        } else {
            File[] files = file.listFiles();
            Arrays.stream(files).parallel().filter(File::isFile).forEach(g::generateFromFile);
        }

        g.normalize();
        g.getProbability().entrySet().stream().sorted((o1, o2) -> o2.getValue().compareTo(o1.getValue())).forEach(System.out::println);
    }


}
