# LangGraph

LangGraph is a framework for building AI apps as a graph of connected steps.

## Why LangGraph?

Use LangGraph when you want an AI app that is:
- predictable
- easy to control
- able to keep state
- good for multi-step workflows
- able to remember things during and across conversations

It is especially useful when a simple single prompt is not enough.

## What Is LangGraph?

LangGraph lets you build an application as nodes and edges.

- **Nodes** are the steps or functions that do work.
- **Edges** decide which node runs next.

This makes your AI flow easier to understand, debug, and extend.

## Core Concepts

### Nodes
Nodes are the building blocks of a graph.  
Each node does one job, such as:
- calling an LLM
- checking a result
- saving data
- deciding the next step

### Edges
Edges connect nodes.  
They define the flow of the application.

- A normal edge goes from one step to the next.
- A conditional edge chooses the next step based on state or logic.

### State
State is the shared data passed through the graph.

It can hold things like:
- user messages
- model responses
- tool outputs
- decisions made by the graph

State is what lets each node know what happened before.

### Checkpoints
Checkpoints save the graph state while it runs.

They help you:
- pause and continue later
- recover from failures
- inspect what happened
- keep conversation context

### Configs
Configs are runtime settings you pass to the graph.

They can control things like:
- thread or session ID
- user ID
- model settings
- app-specific options

Configs help the same graph behave differently for different users or runs.

## Memory

### Short-Term Memory
Short-term memory is the context kept inside the current run or conversation.

In LangGraph, this usually means the current state and checkpoints.

### Long-Term Memory
Long-term memory is information saved outside the current run so it can be used later.

This is often done with a **store**.

### Store
A store is a persistent place to save data across sessions.


You can use it to keep:
- user preferences
- facts about the user
- conversation history summaries
- saved notes or records

Short-term memory helps the graph remember the current conversation.  
Long-term memory helps it remember across many conversations.

## Simple Mental Model

Think of LangGraph like this:

1. A user message enters the graph.
2. A node processes it.
3. An edge decides the next node.
4. State carries the information forward.
5. Checkpoints save progress.
6. A store keeps long-term memory.

## In One Line

LangGraph is a way to build AI apps as controlled, stateful workflows with memory.
