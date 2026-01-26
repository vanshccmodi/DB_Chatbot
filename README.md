---
title: OnceDataBot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🤖 OnceDataBot

A production-grade, **schema-agnostic chatbot** that connects to **any** database (MySQL, PostgreSQL, or SQLite) and provides intelligent querying through **RAG** (Retrieval-Augmented Generation) and **Text-to-SQL**.

**🆓 Powered by Groq for FREE LLM inference!**

## 🌟 Features

- **Multi-Database Support**: Works with **MySQL**, **PostgreSQL**, and **SQLite**
- **Schema-Agnostic**: Works with ANY database schema - no hardcoding required
- **Dynamic Introspection**: Automatically discovers tables, columns, and relationships
- **Hybrid Query Routing**: Intelligently routes queries to RAG or SQL based on intent
- **Semantic Search (RAG)**: FAISS-based vector search for text content
- **Text-to-SQL**: LLM-powered SQL generation with dialect-specific syntax
- **Security First**: Read-only queries, SQL validation, table whitelisting
- **FREE LLM**: Uses Groq API (free tier) with Llama 3.3, Mixtral, and Gemma models

## 🚀 Getting Started

### 1. Configure Secrets

This Space requires the following secrets to be set in your Hugging Face Space settings:

**Required:**
| Secret Name | Description |
|------------|-------------|
| `GROQ_API_KEY` | Your Groq API key ([Get FREE key](https://console.groq.com)) |

**Database Configuration (choose one):**

#### For MySQL:
| Secret Name | Description |
|------------|-------------|
| `DB_TYPE` | Set to `mysql` |
| `DB_HOST` | MySQL server hostname |
| `DB_PORT` | MySQL port (default: 3306) |
| `DB_DATABASE` | Database name |
| `DB_USERNAME` | Database username |
| `DB_PASSWORD` | Database password |

#### For PostgreSQL:
| Secret Name | Description |
|------------|-------------|
| `DB_TYPE` | Set to `postgresql` |
| `DB_HOST` | PostgreSQL server hostname |
| `DB_PORT` | PostgreSQL port (default: 5432) |
| `DB_DATABASE` | Database name |
| `DB_USERNAME` | Database username |
| `DB_PASSWORD` | Database password |

#### For SQLite:
| Secret Name | Description |
|------------|-------------|
| `DB_TYPE` | Set to `sqlite` |
| `SQLITE_PATH` | Path to SQLite database file |

**Optional:**
| Secret Name | Description | Default |
|------------|-------------|---------|
| `GROQ_MODEL` | Groq model to use | `llama-3.3-70b-versatile` |
| `DB_SSL_CA` | Path to SSL CA certificate | None |

### 2. Connect & Use

1. Click **"Connect & Initialize"** in the sidebar
2. Click **"Index Text Data"** to enable semantic search
3. Start asking questions about your data!

## 💬 Example Queries

**Semantic Search (RAG):**
- "What products are related to electronics?"
- "Tell me about customer feedback on shipping"

**Structured Queries (SQL):**
- "How many orders were placed last month?"
- "Show me the top 10 customers by revenue"

**Hybrid:**
- "Find customers who complained about delivery and show their order count"

## 🔒 Security

- **Read-Only Transactions**: All queries run in read-only mode
- **SQL Validation**: Only SELECT statements allowed
- **Forbidden Keywords**: INSERT, UPDATE, DELETE, DROP, etc. are blocked
- **Table Whitelisting**: Only discovered tables are queryable
- **Automatic LIMIT**: All queries have LIMIT clauses enforced

## 🆓 Why Groq?

[Groq](https://console.groq.com) provides **FREE API access** with incredibly fast inference:
- **Llama 3.3 70B** - Best quality, state-of-the-art
- **Llama 3.1 8B Instant** - Fastest responses
- **Mixtral 8x7B** - Great for code and SQL
- **Gemma 2 9B** - Google's efficient model

## 📝 License

MIT License
