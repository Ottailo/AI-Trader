# AI Trader - Automated Trading Platform

A production-grade automated algorithmic trading platform built with Python/FastAPI backend and React/Next.js frontend.

## 🚀 Features

- **Real-time Market Data**: Fetch and process live market data from Alpha Vantage/Polygon.io
- **AI-Powered Predictions**: Deep learning models for trading signal generation
- **Automated Execution**: Secure integration with brokerage APIs (Alpaca)
- **Risk Management**: Dynamic stop-loss and take-profit calculations
- **Real-time Dashboard**: Live portfolio monitoring and trade visualization
- **Backtesting Engine**: Historical performance analysis and strategy validation

## 🏗️ Architecture

```
ai-trader/
├── api/                 # FastAPI backend
│   ├── app/            # Application code
│   ├── migrations/     # Database migrations
│   └── tests/          # Backend tests
├── web/                # Next.js frontend
│   ├── components/     # React components
│   ├── pages/          # Next.js pages
│   └── tests/          # Frontend tests
└── docker-compose.yml  # Development environment
```

## 🛠️ Technology Stack

### Backend

- **Python 3.11+** with FastAPI
- **PostgreSQL** with SQLAlchemy ORM
- **Alembic** for database migrations
- **PyTorch** for machine learning models
- **WebSockets** for real-time communication

### Frontend

- **Next.js 14** with TypeScript
- **React 18** with functional components
- **Tailwind CSS** for styling
- **Zustand** for state management
- **Lightweight Charts** for data visualization

## 📋 Prerequisites

- Node.js 18+ and pnpm 8+
- Python 3.11+
- PostgreSQL 14+
- Docker and Docker Compose (for development)

## 🚀 Quick Start

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd ai-trader
pnpm install
```

### 2. Environment Setup

Create environment files:

```bash
# Backend
cp api/.env.example api/.env

# Frontend
cp web/.env.example web/.env
```

Configure your environment variables:

- Database connection string
- Alpha Vantage/Polygon.io API keys
- Alpaca trading API keys
- JWT secret for authentication

### 3. Database Setup

```bash
# Start PostgreSQL (if using Docker)
docker-compose up -d postgres

# Run migrations
cd api
alembic upgrade head
```

### 4. Start Development Servers

```bash
# Start all services
pnpm dev

# Or start individually
pnpm --filter api dev
pnpm --filter web dev
```

The application will be available at:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🧪 Testing

```bash
# Run all tests
pnpm test

# Run backend tests
pnpm --filter api test

# Run frontend tests
pnpm --filter web test
```

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Production build
docker-compose -f docker-compose.prod.yml up --build
```

## 📊 Project Phases

1. **Phase 1**: Core Infrastructure & Backend Setup ✅
2. **Phase 2**: Machine Learning Engine & Backtesting
3. **Phase 3**: Live Trading & Broker Integration
4. **Phase 4**: Frontend UI & Real-time Dashboard
5. **Phase 5**: Deployment & Operations

## 🔒 Security

- All API keys are encrypted and stored securely
- Environment variables for sensitive configuration
- Input validation and sanitization
- Rate limiting on API endpoints
- CORS configuration for frontend-backend communication

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions, please open an issue in the repository.
