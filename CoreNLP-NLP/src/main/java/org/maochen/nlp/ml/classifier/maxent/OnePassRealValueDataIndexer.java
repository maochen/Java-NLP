package org.maochen.nlp.ml.classifier.maxent;
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import opennlp.model.ComparableEvent;
import opennlp.model.DataIndexer;
import opennlp.model.Event;

import org.maochen.nlp.ml.classifier.maxent.eventstream.EventStream;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An indexer for maxent model data which handles cutoffs for uncommon contextual predicates and
 * provides a unique integer index for each of the predicates and maintains event values.
 */
public class OnePassRealValueDataIndexer implements DataIndexer {

    private float[][] values;

    private int numEvents;
    /**
     * The integer contexts associated with each unique event.
     */
    protected int[][] contexts;

    /**
     * The integer outcome associated with each unique event.
     */
    protected int[] outcomeList;
    /**
     * The names of the outcomes.
     */
    protected String[] outcomeLabels;

    /**
     * The number of times an event occured in the training data.
     */
    protected int[] numTimesEventsSeen;

    /**
     * The predicate/context names.
     */
    protected String[] predLabels;
    /**
     * The number of times each predicate occured.
     */
    protected int[] predCounts;

    public OnePassRealValueDataIndexer(EventStream eventStream, int cutoff, boolean sort) {
        Map<String, Integer> predicateIndex = new HashMap<>();

        System.out.println("Indexing events using cutoff of " + cutoff + "\n");

        System.out.print("\tComputing event counts...  ");
        LinkedList<Event> events = computeEventCounts(eventStream, predicateIndex, cutoff);
        System.out.println("done. " + events.size() + " events");

        System.out.print("\tIndexing...  ");
        List<ComparableEvent> eventsToCompare = index(events, predicateIndex);
        System.out.println("done.");

        System.out.print("Sorting and merging events... ");
        sortAndMerge(eventsToCompare, sort);
        System.out.println("Done indexing.");
    }

    /**
     * Reads events from <tt>eventStream</tt> into a linked list. The predicates associated with
     * each event are counted and any which occur at least <tt>cutoff</tt> times are added to the
     * <tt>predicatesInOut</tt> map along with a unique integer index.
     *
     * @param eventStream     an <code>EventStream</code> value
     * @param predicatesInOut a <code>TObjectIntHashMap</code> value
     * @param cutoff          an <code>int</code> value
     * @return a <code>TLinkedList</code> value
     */
    private LinkedList<Event> computeEventCounts(EventStream eventStream, Map<String, Integer> predicatesInOut, int cutoff) {
        Set<String> predicateSet = new HashSet<>();
        Map<String, Integer> counter = new HashMap<>();
        LinkedList<Event> events = new LinkedList<>();
        while (eventStream.hasNext()) {
            Event ev = eventStream.next();
            events.addLast(ev);
            update(ev.getContext(), predicateSet, counter, cutoff);
        }
        predCounts = new int[predicateSet.size()];
        int index = 0;
        for (Iterator<String> pi = predicateSet.iterator(); pi.hasNext(); index++) {
            String predicate = pi.next();
            predCounts[index] = counter.get(predicate);
            predicatesInOut.put(predicate, index);
        }
        return events;
    }

    /**
     * Updates the set of predicated and counter with the specified event contexts and cutoff.
     *
     * @param ec           The contexts/features which occur in a event.
     * @param predicateSet The set of predicates which will be used for model building.
     * @param counter      The predicate counters.
     * @param cutoff       The cutoff which determines whether a predicate is included.
     */
    protected static void update(String[] ec, Set<String> predicateSet, Map<String, Integer> counter, int cutoff) {
        for (String s : ec) {
            Integer val = counter.get(s);
            val = val == null ? 1 : val + 1;
            counter.put(s, val);

            if (!predicateSet.contains(s) && counter.get(s) >= cutoff) {
                predicateSet.add(s);
            }
        }
    }

    protected int sortAndMerge(List<ComparableEvent> eventsToCompare, boolean sort) {
        int numUniqueEvents = 1;
        numEvents = eventsToCompare.size();
        if (numEvents <= 1) {
            return numUniqueEvents; // nothing to do; edge case (see assertion)
        }

        if (sort) {
            eventsToCompare.sort(ComparableEvent::compareTo);

            ComparableEvent ce = eventsToCompare.get(0);
            for (int i = 1; i < numEvents; i++) {
                ComparableEvent ce2 = eventsToCompare.get(i);

                if (ce.compareTo(ce2) == 0) {
                    ce.seen++; // increment the seen count
                    eventsToCompare.set(i, null); // kill the duplicate
                } else {
                    ce = ce2; // a new champion emerges...
                    numUniqueEvents++; // increment the # of unique events
                }
            }
            System.out.println("done. Reduced " + numEvents + " events to " + numUniqueEvents + ".");
        } else {
            numUniqueEvents = eventsToCompare.size();
        }

        contexts = new int[numUniqueEvents][];
        outcomeList = new int[numUniqueEvents];
        numTimesEventsSeen = new int[numUniqueEvents];

        for (int i = 0, j = 0; i < numEvents; i++) {
            ComparableEvent evt = eventsToCompare.get(i);
            if (null == evt) {
                continue; // this was a dupe, skip over it.
            }
            numTimesEventsSeen[j] = evt.seen;
            outcomeList[j] = evt.outcome;
            contexts[j] = evt.predIndexes;
            ++j;
        }

        values = new float[numUniqueEvents][];
        for (int i = 0, j = 0; i < numEvents; i++) {
            ComparableEvent evt = eventsToCompare.get(i);
            if (null == evt) {
                continue; // this was a dupe, skip over it.
            }
            values[j++] = evt.values;
        }
        return numUniqueEvents;
    }

    /**
     * Utility method for creating a String[] array from a map whose keys are labels (Strings) to be
     * stored in the array and whose values are the indices (Integers) at which the corresponding
     * labels should be inserted.
     *
     * @param labelToIndexMap a <code>TObjectIntHashMap</code> value
     * @return a <code>String[]</code> value
     * @since maxent 1.2.6
     */
    protected static String[] toIndexedStringArray(Map<String, Integer> labelToIndexMap) {
        final String[] array = new String[labelToIndexMap.size()];
        for (String label : labelToIndexMap.keySet()) {
            array[labelToIndexMap.get(label)] = label;
        }
        return array;
    }

    protected List<ComparableEvent> index(LinkedList<Event> events, Map<String, Integer> predicateIndex) {
        Map<String, Integer> omap = new HashMap<>();

        int numEvents = events.size();
        int outcomeCount = 0;
        List<ComparableEvent> eventsToCompare = new ArrayList<>(numEvents);
        List<Integer> indexedContext = new ArrayList<>();

        for (int eventIndex = 0; eventIndex < numEvents; eventIndex++) {
            Event ev = events.removeFirst();
            String[] econtext = ev.getContext();
            ComparableEvent ce;

            int ocID;
            String oc = ev.getOutcome();

            if (omap.containsKey(oc)) {
                ocID = omap.get(oc);
            } else {
                ocID = outcomeCount++;
                omap.put(oc, ocID);
            }

            for (String pred : econtext) {
                if (predicateIndex.containsKey(pred)) {
                    indexedContext.add(predicateIndex.get(pred));
                }
            }

            //drop events with no active features
            if (indexedContext.size() > 0) {
                int[] cons = new int[indexedContext.size()];
                for (int ci = 0; ci < cons.length; ci++) {
                    cons[ci] = indexedContext.get(ci);
                }
                ce = new ComparableEvent(ocID, cons, ev.getValues());
                eventsToCompare.add(ce);
            } else {
                System.err.println("Dropped event " + ev.getOutcome() + ":" + Arrays.asList(ev.getContext()));
            }
//    recycle the TIntArrayList
            indexedContext.clear();
        }
        outcomeLabels = toIndexedStringArray(omap);
        predLabels = toIndexedStringArray(predicateIndex);
        return eventsToCompare;
    }

    @Override
    public int getNumEvents() {
        return numEvents;
    }

    @Override
    public int[][] getContexts() {
        return contexts;
    }

    @Override
    public int[] getNumTimesEventsSeen() {
        return numTimesEventsSeen;
    }

    @Override
    public int[] getOutcomeList() {
        return outcomeList;
    }

    @Override
    public String[] getPredLabels() {
        return predLabels;
    }

    @Override
    public int[] getPredCounts() {
        return predCounts;
    }

    @Override
    public String[] getOutcomeLabels() {
        return outcomeLabels;
    }

    @Override
    public float[][] getValues() {
        return values;
    }

}
