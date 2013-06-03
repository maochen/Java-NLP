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

package classifier.maxent.gis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
 * 
 * @author Tom Morton
 */
public class DataIndexer {

    /**
     * A maxent event representation which we can use to sort based on the predicates indexes
     * contained in the events.
     * 
     * @author Jason Baldridge
     * @version $Revision: 1.2 $, $Date: 2010/09/06 08:02:18 $
     */
    public class ComparableEvent implements Comparable<ComparableEvent> {
        public int outcome;
        public int[] predIndexes;
        public int seen = 1; // the number of times this event
                             // has been seen.

        public float[] values;

        public ComparableEvent(int oc, int[] pids, float[] values) {
            outcome = oc;
            if (values == null) {
                Arrays.sort(pids);
            }
            else {
                sort(pids, values);
            }
            this.values = values; // needs to be sorted like pids
            predIndexes = pids;
        }

        public int compareTo(ComparableEvent o) {
            ComparableEvent ce = o;
            if (outcome < ce.outcome) return -1;
            else if (outcome > ce.outcome) return 1;

            int smallerLength = (predIndexes.length > ce.predIndexes.length ? ce.predIndexes.length
                    : predIndexes.length);

            for (int i = 0; i < smallerLength; i++) {
                if (predIndexes[i] < ce.predIndexes[i]) return -1;
                else if (predIndexes[i] > ce.predIndexes[i]) return 1;
                if (values != null && ce.values != null) {
                    if (values[i] < ce.values[i]) return -1;
                    else if (values[i] > ce.values[i]) return 1;
                }
                else if (values != null) {
                    if (values[i] < 1) return -1;
                    else if (values[i] > 1) return 1;
                }
                else if (ce.values != null) {
                    if (1 < ce.values[i]) return -1;
                    else if (1 > ce.values[i]) return 1;
                }
            }

            if (predIndexes.length < ce.predIndexes.length) return -1;
            else if (predIndexes.length > ce.predIndexes.length) return 1;

            return 0;
        }

        public String toString() {
            StringBuffer s = new StringBuffer().append(outcome).append(":");
            for (int i = 0; i < predIndexes.length; i++) {
                s.append(" ").append(predIndexes[i]);
                if (values != null) {
                    s.append("=").append(values[i]);
                }
            }
            return s.toString();
        }

        private void sort(int[] pids, float[] values) {
            for (int mi = 0; mi < pids.length; mi++) {
                int min = mi;
                for (int pi = mi + 1; pi < pids.length; pi++) {
                    if (pids[min] > pids[pi]) {
                        min = pi;
                    }
                }
                int pid = pids[mi];
                pids[mi] = pids[min];
                pids[min] = pid;
                float val = values[mi];
                values[mi] = values[min];
                values[min] = val;
            }
        }
    }

    /**
     * The context of a decision point during training. This includes contextual predicates and an
     * outcome.
     * 
     * @author Jason Baldridge
     * @version $Revision: 1.2 $, $Date: 2010/09/06 08:02:18 $
     */
    static class Event {
        private String outcome;
        private String[] context;
        private float[] values;

        public Event(String outcome, String[] context, float[] values) {
            this.outcome = outcome;
            this.context = context;
            this.values = values;
        }

        public String toString() {
            StringBuffer sb = new StringBuffer();
            sb.append(outcome).append(" [");
            if (context.length > 0) {
                sb.append(context[0]);
                if (values != null) {
                    sb.append("=" + values[0]);
                }
            }
            for (int ci = 1; ci < context.length; ci++) {
                sb.append(" ").append(context[ci]);
                if (values != null) {
                    sb.append("=" + values[ci]);
                }
            }
            sb.append("]");
            return sb.toString();
        }

    }

    private int numEvents;

    /** The predicate/context names. */
    protected String[] predLabels;

    /** The integer outcome associated with each unique event. */
    protected int[] outcomeList;

    /** The names of the outcomes. */
    protected String[] outcomeLabels;

    protected int[] numTimesEventsSeen;

    /** The integer contexts associated with each unique event. */
    int[][] contexts;

    /** The number of times each predicate occured. */
    protected int[] predCounts;

    float[][] values;

    /**
     * Updates the set of predicated and counter with the specified event contexts and cutoff.
     * 
     * @param ec The contexts/features which occur in a event.
     * @param predicateSet The set of predicates which will be used for model building.
     * @param counter The predicate counters.
     * @param cutoff The cutoff which determines whether a predicate is included.
     */
    protected static void update(String[] ec, Set<String> predicateSet, Map<String, Integer> counter, int cutoff) {
        for (int j = 0; j < ec.length; j++) {
            Integer i = counter.get(ec[j]);
            if (i == null) {
                counter.put(ec[j], 1);
            }
            else {
                counter.put(ec[j], i + 1);
            }
            if (!predicateSet.contains(ec[j]) && counter.get(ec[j]) >= cutoff) {
                predicateSet.add(ec[j]);
            }
        }
    }

    // NEEDS TO CHANGE

    /**
     * Parses the specified contexts and re-populates context array with features and returns the
     * values for these features. If all values are unspecified, then null is returned.
     * 
     * @param contexts The contexts with real values specified.
     * @return The value for each context or null if all values are unspecified.
     */
    public static float[] parseContexts(String[] contexts) {
        boolean hasRealValue = false;
        float[] values = new float[contexts.length];
        for (int ci = 0; ci < contexts.length; ci++) {
            int ei = contexts[ci].lastIndexOf("=");
            if (ei > 0 && ei + 1 < contexts[ci].length()) {
                boolean gotReal = true;
                try {
                    values[ci] = Float.parseFloat(contexts[ci].substring(ei + 1));
                } catch (NumberFormatException e) {
                    gotReal = false;
                    System.err.println("Unable to determine value in context:" + contexts[ci]);
                    values[ci] = 1;
                }
                if (gotReal) {
                    if (values[ci] < 0) {
                        throw new RuntimeException("Negitive values are not allowed: " + contexts[ci]);
                    }
                    contexts[ci] = contexts[ci].substring(0, ei);
                    hasRealValue = true;
                }
            }
            else {
                values[ci] = 1;
            }
        }
        if (!hasRealValue) {
            values = null;
        }
        return values;
    }

    private Event createEvent(String obs) {
        int lastSpace = obs.lastIndexOf(' ');
        if (lastSpace == -1) return null;
        else {
            String[] contexts = obs.substring(0, lastSpace).split("\\s+");
            float[] values = parseContexts(contexts);
            return new Event(obs.substring(lastSpace + 1), contexts, values);
        }
    }

    // ----------

    /**
     * Two argument constructor for DataIndexer.
     * 
     * @param trainingData An Event[] which contains the a list of all the Events seen in the
     *            training data.
     * @param cutoff The minimum number of times a predicate must have been observed in order to be
     *            included in the model.
     */
    public DataIndexer(List<String[]> trainingData, int cutoff) {
        Map<String, Integer> predicateIndex = new HashMap<String, Integer>();
        System.out.println("Indexing events using cutoff of " + cutoff + "\n");

        System.out.print("\tComputing event counts...  ");
        LinkedList<Event> events = computeEventCounts(trainingData, predicateIndex, cutoff);
        System.out.println("done. " + events.size() + " events");

        System.out.print("\tIndexing...  ");
        List<Event> eventsToCompare = index(events, predicateIndex);

        // done with event list
        events = null;
        // done with predicates
        predicateIndex = null;

        System.out.println("done.");

        System.out.print("Sorting and merging events... ");
        sortAndMerge(eventsToCompare, true);
        System.out.println("Done indexing.");
    }

    /**
     * Reads events from <tt>eventStream</tt> into a linked list. The predicates associated with
     * each event are counted and any which occur at least <tt>cutoff</tt> times are added to the
     * <tt>predicatesInOut</tt> map along with a unique integer index.
     * 
     * @param eventStream an <code>EventStream</code> value
     * @param predicatesInOut a <code>TObjectIntHashMap</code> value
     * @param cutoff an <code>int</code> value
     * @return a <code>TLinkedList</code> value
     */
    private LinkedList<Event> computeEventCounts(List<String[]> trainingData, Map<String, Integer> predicatesInOut,
            int cutoff) {
        Set<String> predicateSet = new HashSet<String>();
        Map<String, Integer> counter = new HashMap<String, Integer>();
        LinkedList<Event> events = new LinkedList<Event>();

        for (String[] entry : trainingData) {

            StringBuffer buf = new StringBuffer();
            for (String s : entry) {
                buf.append(s);
                buf.append(" ");
            }

            Event ev = createEvent(buf.toString().trim());

            // --------------
            System.out.print("Context: " + Arrays.toString(ev.context));
            System.out.print("| OutCome: " + ev.outcome);
            System.out.println("| Value: " + Arrays.toString(ev.values));

            // ------------

            events.addLast(ev);
            update(ev.context, predicateSet, counter, cutoff);
        }

        predCounts = new int[predicateSet.size()];
        int index = 0;
        for (Iterator<String> pi = predicateSet.iterator(); pi.hasNext(); index++) {
            String predicate = (String) pi.next();
            predCounts[index] = counter.get(predicate);
            predicatesInOut.put(predicate, index);
        }
        return events;
    }

    /**
     * Sorts and uniques the array of comparable events and return the number of unique events. This
     * method will alter the eventsToCompare array -- it does an in place sort, followed by an in
     * place edit to remove duplicates.
     * 
     * @param eventsToCompare a <code>ComparableEvent[]</code> value
     * @return The number of unique events in the specified list.
     * @since maxent 1.2.6
     */
    protected int sortAndMerge(List eventsToCompare, boolean sort) {
        int numUniqueEvents = 1;
        numEvents = eventsToCompare.size();
        if (sort) {
            Collections.sort(eventsToCompare);
            if (numEvents <= 1) {
                return numUniqueEvents; // nothing to do; edge case (see assertion)
            }

            ComparableEvent ce = (ComparableEvent) eventsToCompare.get(0);
            for (int i = 1; i < numEvents; i++) {
                ComparableEvent ce2 = (ComparableEvent) eventsToCompare.get(i);

                if (ce.compareTo(ce2) == 0) {
                    ce.seen++; // increment the seen count
                    eventsToCompare.set(i, null); // kill the duplicate
                }
                else {
                    ce = ce2; // a new champion emerges...
                    numUniqueEvents++; // increment the # of unique events
                }
            }
        }
        else {
            numUniqueEvents = eventsToCompare.size();
        }
        if (sort) System.out.println("done. Reduced " + numEvents + " events to " + numUniqueEvents + ".");

        contexts = new int[numUniqueEvents][];
        outcomeList = new int[numUniqueEvents];
        numTimesEventsSeen = new int[numUniqueEvents];

        for (int i = 0, j = 0; i < numEvents; i++) {
            ComparableEvent evt = (ComparableEvent) eventsToCompare.get(i);
            if (null == evt) {
                continue; // this was a dupe, skip over it.
            }
            numTimesEventsSeen[j] = evt.seen;
            outcomeList[j] = evt.outcome;
            contexts[j] = evt.predIndexes;
            ++j;
        }

        values = new float[numUniqueEvents][];
        int numEvents = eventsToCompare.size();
        for (int i = 0, j = 0; i < numEvents; i++) {
            ComparableEvent evt = (ComparableEvent) eventsToCompare.get(i);
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

    protected List<Event> index(LinkedList<Event> events, Map<String, Integer> predicateIndex) {
        Map<String, Integer> omap = new HashMap<String, Integer>();

        int numEvents = events.size();
        int outcomeCount = 0;
        List eventsToCompare = new ArrayList<Event>(numEvents);
        List<Integer> indexedContext = new ArrayList<Integer>();

        for (int eventIndex = 0; eventIndex < numEvents; eventIndex++) {
            Event ev = (Event) events.removeFirst();
            String[] econtext = ev.context;
            ComparableEvent ce;

            int ocID;
            String oc = ev.outcome;

            if (omap.containsKey(oc)) {
                ocID = omap.get(oc);
            }
            else {
                ocID = outcomeCount++;
                omap.put(oc, ocID);
            }

            for (int i = 0; i < econtext.length; i++) {
                String pred = econtext[i];
                if (predicateIndex.containsKey(pred)) {
                    indexedContext.add(predicateIndex.get(pred));
                }
            }

            // drop events with no active features
            if (indexedContext.size() > 0) {
                int[] cons = new int[indexedContext.size()];
                for (int ci = 0; ci < cons.length; ci++) {
                    cons[ci] = indexedContext.get(ci);
                }
                ce = new ComparableEvent(ocID, cons, ev.values);
                eventsToCompare.add(ce);
            }
            else {
                System.err.println("Dropped event " + ev.outcome + ":" + Arrays.asList(ev.context));
            }
            // recycle the TIntArrayList
            indexedContext.clear();
        }
        outcomeLabels = toIndexedStringArray(omap);
        predLabels = toIndexedStringArray(predicateIndex);
        return eventsToCompare;
    }

}
