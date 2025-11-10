#pragma once
#include <memory>

#ifdef DEBUG_TRACER
// Use TRACE_DECL for declarations that must live in the enclosing scope
#define TRACE_DECL(...) __VA_ARGS__
// Use TRACE_STMT for statements; wrapped to be safe in if/else contexts
#define TRACE_STMT(...) \
    do                  \
    {                   \
        __VA_ARGS__     \
    } while (0)
// Backwards-compatible default: TRACE -> statement-style
#define TRACE(...) TRACE_STMT(__VA_ARGS__)
#define TRACE_PARAM , std::shared_ptr<tracer_t> ptracer
#define TRACE_ARG , ptracer
#define TRACE_MEMBER_DECL std::shared_ptr<tracer_t> ptracer;
#define TRACE_MEMBER_INIT , ptracer(ptracer)
#define TRACE_MEMBER_MOVE_INIT , ptracer(other.ptracer)
#define TRACE_MEMBER_MOVE_INIT_OTHER(ptr) , ptracer(ptr)
#define TRACE_GET_PTRACER \
    std::shared_ptr<tracer_t> get_ptracer() const { return ptracer; }
#define WHILE_TRACER(ptr) while ((ptr) ? (ptr)->poll_for_closed_event() : false)
#else
#define TRACE(...)
#define TRACE_PARAM
#define TRACE_ARG
#define TRACE_MEMBER_DECL
#define TRACE_MEMBER_INIT
#define TRACE_MEMBER_MOVE_INIT
#define TRACE_MEMBER_MOVE_INIT_OTHER(ptr)
#define TRACE_GET_PTRACER
#define WHILE_TRACER(ptr) while (!stopp.load())
#endif
