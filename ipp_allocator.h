// $Id$

#ifndef _SSE_IPP_ALLOCATOR_H_
#define _SSE_IPP_ALLOCATOR_H_

#include <limits>
#include <stdexcept>

#include "sse_ipp.h"

namespace ipp
{

	template<typename T>
	class Allocator
	{
	public:
		typedef T value_type;
		typedef value_type * pointer;
		typedef const value_type * const_pointer;
		typedef value_type& reference;
		typedef const value_type& const_reference;
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;

		template<typename U>
		struct rebind
		{
			typedef Allocator<U> other;
		};

		explicit Allocator()
		{}

		~Allocator()
		{}

		explicit Allocator(Allocator const&)
		{}

		template<typename U>
		explicit Allocator(Allocator<U> const&)
		{}

		pointer address(reference r) { return &r; }
		const_pointer address(const_reference r) { return &r; }

		pointer allocate(size_type count, typename std::allocator<void>::const_pointer = 0)
		{
			pointer ptr = reinterpret_cast<pointer>( ipp::malloc<T>(count * sizeof(T)) );
			if (! ptr)
				throw std::bad_alloc();
			return ptr;
		}

		void deallocate(pointer p, size_type)
		{
			if (p)
				ipp::free(p);
		}

		size_type max_size() const
		{
			return std::numeric_limits<size_type>::max() / sizeof(T);
		}

		void construct(pointer p, const T& t) { new(p) T(t); }
		void destroy(pointer p) { p->~T(); }

		bool operator == (Allocator const&) { return true; }
		bool operator != (Allocator const& a) { return ! operator == (a); }
	};

}

#endif
